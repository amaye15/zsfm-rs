use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::config::TiRexConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct Block {
    norm_slstm: Vec<f32>,     // [D]
    fgate_w: Vec<f32>,        // [NH, DH, DH] — LinearHeadwiseExpand weight
    igate_w: Vec<f32>,
    zgate_w: Vec<f32>,
    ogate_w: Vec<f32>,
    slstm_kernel: Vec<f32>,   // [NH, DH, NG*DH] = [4, 128, 512]
    slstm_bias: Vec<f32>,     // [NG*NH*DH = 2048] in [NG, NH, DH] order
    group_norm_w: Vec<f32>,   // [D] (learned offset, applied as 1 + w)
    norm_ffn: Vec<f32>,       // [D]
    ffn_gate_w: Tensor,       // [UP, D]
    ffn_up_w: Tensor,         // [UP, D]
    ffn_down_w: Tensor,       // [D, UP]
}

struct EmbedBlock {
    hidden_w: Tensor,   // [H_DIM, IN_DIM]
    hidden_b: Vec<f32>, // [H_DIM]
    output_w: Tensor,   // [OUT_DIM, H_DIM]
    output_b: Vec<f32>, // [OUT_DIM]
    residual_w: Tensor, // [OUT_DIM, IN_DIM]
    residual_b: Vec<f32>, // [OUT_DIM]
}

pub struct TiRexModel {
    device: Device,
    config: TiRexConfig,
    in_emb: EmbedBlock,
    blocks: Vec<Block>,
    out_norm: Vec<f32>, // [D]
    out_emb: EmbedBlock,
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

fn load_t(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    let qt = content
        .tensor(reader, name, device)
        .with_context(|| format!("tensor '{name}' not found in GGUF"))?;
    Ok(qt.dequantize(device)?.to_dtype(DType::F32)?)
}

fn load_vec(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Vec<f32>> {
    Ok(load_t(content, reader, name, device)?.flatten_all()?.to_vec1()?)
}

fn load_embed_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<EmbedBlock> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(EmbedBlock {
        hidden_w:   load_t(&content, reader, &p("hidden.weight"), device)?,
        hidden_b:   load_vec(&content, reader, &p("hidden.bias"), device)?,
        output_w:   load_t(&content, reader, &p("output.weight"), device)?,
        output_b:   load_vec(&content, reader, &p("output.bias"), device)?,
        residual_w: load_t(&content, reader, &p("residual.weight"), device)?,
        residual_b: load_vec(&content, reader, &p("residual.bias"), device)?,
    })
}

impl TiRexModel {
    pub fn load(gguf_path: &Path, config: TiRexConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let in_emb = load_embed_block(&content, &mut reader, "in_emb", &device)?;

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for n in 0..config.num_blocks {
            let p = |s: &str| format!("blk.{n}.{s}");
            blocks.push(Block {
                norm_slstm:   load_vec(&content, &mut reader, &p("norm_slstm"), &device)?,
                fgate_w:      load_vec(&content, &mut reader, &p("fgate.weight"), &device)?,
                igate_w:      load_vec(&content, &mut reader, &p("igate.weight"), &device)?,
                zgate_w:      load_vec(&content, &mut reader, &p("zgate.weight"), &device)?,
                ogate_w:      load_vec(&content, &mut reader, &p("ogate.weight"), &device)?,
                slstm_kernel: load_vec(&content, &mut reader, &p("slstm_kernel"), &device)?,
                slstm_bias:   load_vec(&content, &mut reader, &p("slstm_bias"), &device)?,
                group_norm_w: load_vec(&content, &mut reader, &p("group_norm"), &device)?,
                norm_ffn:     load_vec(&content, &mut reader, &p("norm_ffn"), &device)?,
                ffn_gate_w:   load_t(&content, &mut reader, &p("ffn_gate.weight"), &device)?,
                ffn_up_w:     load_t(&content, &mut reader, &p("ffn_up.weight"), &device)?,
                ffn_down_w:   load_t(&content, &mut reader, &p("ffn_down.weight"), &device)?,
            });
        }

        let out_norm = load_vec(&content, &mut reader, "out_norm", &device)?;
        let out_emb = load_embed_block(&content, &mut reader, "out_emb", &device)?;

        Ok(TiRexModel { device, config, in_emb, blocks, out_norm, out_emb })
    }

    /// Forecast quantiles and mean for a single time series.
    ///
    /// Returns `(quantiles, mean)` where:
    /// - `quantiles`: [prediction_length, num_quantiles] in the config's quantile order
    /// - `mean`: [prediction_length] (median / 0.5 quantile)
    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let cfg = &self.config;
        let patch_size = cfg.patch_size;
        let n_quantiles = cfg.num_quantiles;
        let median_idx = cfg.quantiles.iter().position(|&q| (q - 0.5).abs() < 1e-6).unwrap_or(4);

        // Number of AR steps needed
        let n_steps = prediction_length.div_ceil(patch_size);

        // Accumulate: [n_steps * patch_size, n_quantiles]
        let mut all_q: Vec<f32> = Vec::with_capacity(n_steps * patch_size * n_quantiles);

        // Working context (may grow with NaN placeholders between steps)
        let mut ctx: Vec<f32> = context.to_vec();

        for _ in 0..n_steps {
            // Pad/truncate to train_ctx_len
            let full_ctx = adjust_context(&ctx, cfg.train_ctx_len);

            // StandardScaler
            let (loc, scale) = standard_scaler(&full_ctx);

            // Scale values, zero-out NaN; build mask
            let s_vals: Vec<f32> = full_ctx.iter().map(|&x| {
                if x.is_nan() { 0.0f32 } else { (x - loc) / scale }
            }).collect();
            let s_mask: Vec<f32> = full_ctx.iter().map(|&x| {
                if x.is_nan() { 0.0f32 } else { 1.0f32 }
            }).collect();

            // Patch: [num_patches, patch_size]
            let num_patches = cfg.train_ctx_len / patch_size;
            let mut patched_vals = vec![0.0f32; num_patches * patch_size];
            let mut patched_mask = vec![0.0f32; num_patches * patch_size];
            for p in 0..num_patches {
                let start = p * patch_size;
                patched_vals[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_vals[start..start + patch_size]);
                patched_mask[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_mask[start..start + patch_size]);
            }

            // Concatenate [vals | mask] → [num_patches, 2*patch_size]
            let in_dim = patch_size * 2;
            let mut x_in = vec![0.0f32; num_patches * in_dim];
            for p in 0..num_patches {
                x_in[p * in_dim..p * in_dim + patch_size]
                    .copy_from_slice(&patched_vals[p * patch_size..(p + 1) * patch_size]);
                x_in[p * in_dim + patch_size..p * in_dim + in_dim]
                    .copy_from_slice(&patched_mask[p * patch_size..(p + 1) * patch_size]);
            }

            // ResidualBlock (input patch embedding): [num_patches, in_dim] → [num_patches, D]
            let mut hidden = residual_block_forward(
                &x_in, num_patches, in_dim, cfg.input_ff_dim, cfg.embedding_dim,
                &self.in_emb, &self.device,
            )?;

            // 12 sLSTM blocks
            for block in &self.blocks {
                hidden = self.forward_slstm_block(&hidden, num_patches, block)?;
            }

            // out_norm (RMSNorm)
            rms_norm_inplace(&mut hidden, &self.out_norm, cfg.embedding_dim, 1e-6);

            // output_patch_embedding: [num_patches, D] → [num_patches, Q*P]
            let out_dim = cfg.output_dim();
            let preds = residual_block_forward(
                &hidden, num_patches, cfg.embedding_dim, cfg.input_ff_dim, out_dim,
                &self.out_emb, &self.device,
            )?;
            // preds: [num_patches, Q*patch_size] = [num_patches, 9*32]
            // Take last patch: [Q*patch_size = 288]
            let last = &preds[(num_patches - 1) * out_dim..num_patches * out_dim];

            // Rescale and collect per-quantile for this step
            for q in 0..n_quantiles {
                for d in 0..patch_size {
                    // preds layout: [Q*P] = [q0_p0, q0_p1, ..., q0_p31, q1_p0, ...]
                    // Wait — the Python output is [Q, patch_size] after unflatten(-1, (Q, P))
                    // and output_patch_embedding gives [S, Q*P] where Q varies slowest
                    // i.e., preds[s, q*P + d] = prediction for quantile q, offset d in patch
                    let v = last[q * patch_size + d] * scale + loc;
                    all_q.push(v);
                }
            }

            // Extend context with NaN patch for next AR step
            ctx.extend(std::iter::repeat(f32::NAN).take(patch_size));
        }

        // all_q shape: [n_steps, n_quantiles, patch_size] when indexed as
        //   all_q[step * n_quantiles * patch_size + q * patch_size + d]
        // We want [prediction_length, n_quantiles]
        let total = n_steps * patch_size;
        let mut quantiles: Vec<Vec<f32>> = vec![Vec::with_capacity(prediction_length); n_quantiles];
        for t in 0..prediction_length.min(total) {
            let step = t / patch_size;
            let d = t % patch_size;
            for q in 0..n_quantiles {
                let v = all_q[step * n_quantiles * patch_size + q * patch_size + d];
                quantiles[q].push(v);
            }
        }

        let mean: Vec<f32> = (0..prediction_length).map(|t| quantiles[median_idx][t]).collect();

        Ok((quantiles, mean))
    }

    fn forward_slstm_block(&self, x: &[f32], s: usize, blk: &Block) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let d = cfg.embedding_dim;
        let nh = cfg.num_heads;
        let dh = cfg.head_dim();
        let ng = 4usize;

        // Pre-norm
        let mut x_n = x.to_vec();
        rms_norm_inplace(&mut x_n, &blk.norm_slstm, d, 1e-6);

        // Gate projections (headwise linear, batch over S tokens)
        let f_proj = headwise_linear_batch(&x_n, &blk.fgate_w, s, nh, dh)?;
        let i_proj = headwise_linear_batch(&x_n, &blk.igate_w, s, nh, dh)?;
        let z_proj = headwise_linear_batch(&x_n, &blk.zgate_w, s, nh, dh)?;
        let o_proj = headwise_linear_batch(&x_n, &blk.ogate_w, s, nh, dh)?;

        // Concatenate: x_g[s] = [f_proj[s], i_proj[s], z_proj[s], o_proj[s]] in [NG, NH, DH] order
        let mut x_g = vec![0.0f32; s * ng * d];
        for t in 0..s {
            x_g[t * ng * d..t * ng * d + d].copy_from_slice(&f_proj[t * d..(t + 1) * d]);
            x_g[t * ng * d + d..t * ng * d + 2 * d].copy_from_slice(&i_proj[t * d..(t + 1) * d]);
            x_g[t * ng * d + 2 * d..t * ng * d + 3 * d].copy_from_slice(&z_proj[t * d..(t + 1) * d]);
            x_g[t * ng * d + 3 * d..t * ng * d + 4 * d].copy_from_slice(&o_proj[t * d..(t + 1) * d]);
        }

        // Sequential sLSTM recurrence
        let mut h = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        let mut n = vec![0.0f32; d];
        let mut m = vec![f32::NEG_INFINITY; d];
        let mut h_out = vec![0.0f32; s * d];

        for t in 0..s {
            let wx = &x_g[t * ng * d..(t + 1) * ng * d]; // [NG*D = 2048]
            let ry = compute_ry(&h, &blk.slstm_kernel, nh, dh, ng);
            // raw = wx + ry + bias, all in [NG, NH, DH] order
            let mut raw = vec![0.0f32; ng * d];
            for i in 0..ng * d {
                raw[i] = wx[i] + ry[i] + blk.slstm_bias[i];
            }

            let is_first = n.iter().all(|&v| v == 0.0);
            let mut h_new = vec![0.0f32; d];
            let mut c_new = vec![0.0f32; d];
            let mut n_new = vec![0.0f32; d];
            let mut m_new = vec![0.0f32; d];

            // gate indices in flat [NG, NH, DH] layout:
            //   g=0 (offset 0) → iraw (from fgate module, input gate)
            //   g=1 (offset D) → fraw (from igate module, forget gate)
            //   g=2 (offset 2D) → zraw (cell gate)
            //   g=3 (offset 3D) → oraw (output gate)
            for i in 0..d {
                let iraw = raw[i];
                let fraw = raw[d + i];
                let zraw = raw[2 * d + i];
                let oraw = raw[3 * d + i];

                let log_sigma_f = log_sigmoid(fraw.min(15.0));
                let logfplusm = m[i] + log_sigma_f;
                let mnew = if is_first { iraw } else { iraw.max(logfplusm) };

                let ogate = sigmoid(oraw);
                let igate = (iraw - mnew).exp().min(1.0);
                let fgate = (logfplusm - mnew).exp().min(1.0);
                let zgate = zraw.tanh();

                c_new[i] = fgate * c[i] + igate * zgate;
                n_new[i] = fgate * n[i] + igate;
                // Guard against n≈0 (shouldn't happen in normal operation)
                h_new[i] = if n_new[i].abs() > 1e-8 {
                    ogate * c_new[i] / n_new[i]
                } else {
                    0.0
                };
                m_new[i] = mnew;
            }

            h = h_new;
            c = c_new;
            n = n_new;
            m = m_new;
            h_out[t * d..(t + 1) * d].copy_from_slice(&h);
        }

        // MultiHeadLayerNorm (group norm per head per token) + reshape to [S, D]
        let mut y = vec![0.0f32; s * d];
        for t in 0..s {
            let h_t = &h_out[t * d..(t + 1) * d]; // [D = NH*DH]
            // Group norm: for each head h, normalize [DH] elements, apply (1 + w)
            for head in 0..nh {
                let start = head * dh;
                let end = start + dh;
                let h_slice = &h_t[start..end];
                let mean = h_slice.iter().sum::<f32>() / dh as f32;
                let var = h_slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dh as f32;
                let inv_std = 1.0 / (var + 1e-5f32).sqrt();
                let w_slice = &blk.group_norm_w[start..end];
                for d_i in 0..dh {
                    y[t * d + start + d_i] = (h_t[start + d_i] - mean) * inv_std * (1.0 + w_slice[d_i]);
                }
            }
        }

        // Residual: x + y
        let mut x_out = x.to_vec();
        for i in 0..s * d {
            x_out[i] += y[i];
        }

        // FFN pre-norm
        let mut x_n2 = x_out.clone();
        rms_norm_inplace(&mut x_n2, &blk.norm_ffn, d, 1e-6);

        // FFN: SiLU gated: (silu(gate(x)) * up(x)) → down
        let ffn_out = ffn_forward(&x_n2, s, d, cfg.ffn_up_dim, blk, &self.device)?;

        // Residual: x_out + ffn
        for i in 0..s * d {
            x_out[i] += ffn_out[i];
        }

        Ok(x_out)
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn log_sigmoid(x: f32) -> f32 {
    // log(sigmoid(x)) = -log(1 + exp(-x)) for stability
    if x >= 0.0 {
        -(1.0 + (-x).exp()).ln()
    } else {
        x - (1.0 + x.exp()).ln()
    }
}

/// StandardScaler: compute (loc, scale) ignoring NaN values.
fn standard_scaler(x: &[f32]) -> (f32, f32) {
    let eps = 1e-5f32;
    let valid: Vec<f32> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    if valid.is_empty() {
        return (0.0, 1.0);
    }
    let loc = valid.iter().sum::<f32>() / valid.len() as f32;
    let variance = valid.iter().map(|&v| (v - loc) * (v - loc)).sum::<f32>() / valid.len() as f32;
    let scale = variance.sqrt();
    let scale = if scale == 0.0 { loc.abs() + eps } else { scale };
    (loc, scale)
}

/// Pad (left with NaN) or truncate (from front) context to target_len.
fn adjust_context(x: &[f32], target_len: usize) -> Vec<f32> {
    if x.len() >= target_len {
        x[x.len() - target_len..].to_vec()
    } else {
        let pad = target_len - x.len();
        let mut out = vec![f32::NAN; target_len];
        out[pad..].copy_from_slice(x);
        out
    }
}

/// RMSNorm in-place: x = x / rms(x) * w
fn rms_norm_inplace(x: &mut [f32], w: &[f32], d: usize, eps: f32) {
    let s = x.len() / d;
    for t in 0..s {
        let row = &mut x[t * d..(t + 1) * d];
        let ms = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = *v * scale * w[i];
        }
    }
}

/// LinearHeadwiseExpand: [S, D] → [S, D] using per-head weights [NH, DH, DH].
fn headwise_linear_batch(x: &[f32], w: &[f32], s: usize, nh: usize, dh: usize) -> Result<Vec<f32>> {
    let d = nh * dh;
    // w layout: [NH, DH_out, DH_in] = [NH, DH, DH]
    // For each token t, head h: y[t, h, :] = x[t, h, :] @ w[h].T
    // w[h] has shape [DH_out, DH_in], so w[h].T has shape [DH_in, DH_out]
    let mut out = vec![0.0f32; s * d];
    for t in 0..s {
        for h in 0..nh {
            let x_h = &x[t * d + h * dh..t * d + (h + 1) * dh]; // [DH_in]
            let w_h = &w[h * dh * dh..(h + 1) * dh * dh]; // [DH_out, DH_in]
            let out_h = &mut out[t * d + h * dh..t * d + (h + 1) * dh]; // [DH_out]
            // out_h[o] = sum_d(x_h[d] * w_h[o*DH_in + d])
            for o in 0..dh {
                let mut acc = 0.0f32;
                let w_row = &w_h[o * dh..(o + 1) * dh];
                for (di, &xv) in x_h.iter().enumerate() {
                    acc += xv * w_row[di];
                }
                out_h[o] = acc;
            }
        }
    }
    Ok(out)
}

/// Compute recurrent contribution Ry from h_prev and the sLSTM kernel.
///
/// kernel layout: [NH, DH, NG*DH] — for each head, h_head @ kernel_head → [NG*DH]
/// Output: [NG*NH*DH = 2048] in [NG, NH, DH] order (matching gate projection order).
fn compute_ry(h: &[f32], kernel: &[f32], nh: usize, dh: usize, ng: usize) -> Vec<f32> {
    // ry_raw[NH, NG, DH] — will be permuted to [NG, NH, DH]
    let mut ry_raw = vec![0.0f32; nh * ng * dh];
    for head in 0..nh {
        let h_h = &h[head * dh..(head + 1) * dh]; // [DH]
        // kernel[head] has shape [DH, NG*DH] in row-major
        let k_offset = head * dh * ng * dh;
        for gate_d in 0..ng * dh {
            let mut acc = 0.0f32;
            for di in 0..dh {
                acc += h_h[di] * kernel[k_offset + di * ng * dh + gate_d];
            }
            // gate_d indexes [NG, DH] flat: gate = gate_d/dh, d = gate_d%dh
            ry_raw[head * ng * dh + gate_d] = acc;
        }
    }

    // Permute [NH, NG, DH] → [NG, NH, DH]
    let mut out = vec![0.0f32; nh * ng * dh];
    for head in 0..nh {
        for g in 0..ng {
            for d in 0..dh {
                out[g * nh * dh + head * dh + d] = ry_raw[head * ng * dh + g * dh + d];
            }
        }
    }
    out
}

/// ResidualBlock forward: x → relu(x @ Wh.T + bh) @ Wo.T + bo + x @ Wr.T + br
fn residual_block_forward(
    x: &[f32],
    s: usize,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
    emb: &EmbedBlock,
    device: &Device,
) -> Result<Vec<f32>> {
    // x: [S, in_dim]
    let xt = Tensor::from_slice(x, (s, in_dim), device)?;

    // hidden: relu(x @ Wh.T + bh)
    let h = xt.matmul(&emb.hidden_w.t()?)
        .with_context(|| "in_emb hidden matmul")?;
    // Add bias
    let bh = Tensor::from_slice(&emb.hidden_b, (1, h_dim), device)?;
    let h = h.broadcast_add(&bh)?;
    let h = h.relu()?;

    // output: h @ Wo.T + bo
    let out = h.matmul(&emb.output_w.t()?)
        .with_context(|| "in_emb output matmul")?;
    let bo = Tensor::from_slice(&emb.output_b, (1, out_dim), device)?;
    let out = out.broadcast_add(&bo)?;

    // residual: x @ Wr.T + br
    let res = xt.matmul(&emb.residual_w.t()?)
        .with_context(|| "in_emb residual matmul")?;
    let br = Tensor::from_slice(&emb.residual_b, (1, out_dim), device)?;
    let res = res.broadcast_add(&br)?;

    Ok((out + res)?.flatten_all()?.to_vec1()?)
}

/// FFN: SiLU(gate(x)) * up(x) → down
fn ffn_forward(x: &[f32], s: usize, in_dim: usize, up_dim: usize, blk: &Block, device: &Device) -> Result<Vec<f32>> {
    let xt = Tensor::from_slice(x, (s, in_dim), device)?;

    // gate proj: [S, up_dim]
    let gate = xt.matmul(&blk.ffn_gate_w.t()?)?;
    // up proj: [S, up_dim]
    let up = xt.matmul(&blk.ffn_up_w.t()?)?;
    // SiLU gated: silu(gate) * up
    let h = gate.silu()?.mul(&up)?;
    // down proj: [S, in_dim]
    let out = h.matmul(&blk.ffn_down_w.t()?)?;

    Ok(out.flatten_all()?.to_vec1()?)
}
