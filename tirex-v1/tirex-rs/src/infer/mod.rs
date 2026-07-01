use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use simdeez::prelude::*;
use simdeez::math::{SimdMathF32Core, SimdMathF32Hyperbolic};

use crate::config::TiRexConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct Block {
    norm_slstm: Vec<f32>,      // [D]
    fizo_w: Vec<f32>,          // [NH, 4*DH, DH] — f,i,z,o gates fused at load time
    slstm_kernel_t: Vec<f32>,  // [NH, NG*DH, DH] = [4, 512, 128] — transposed for SIMD dot
    slstm_bias: Vec<f32>,      // [NG*NH*DH = 2048] in [NG, NH, DH] order
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

        let nh  = config.num_heads;
        let dh  = config.head_dim();
        let wpp = dh * dh; // weights per head per gate

        let mut blocks = Vec::with_capacity(config.num_blocks);
        for n in 0..config.num_blocks {
            let p = |s: &str| format!("blk.{n}.{s}");

            // Load the 4 gate weights, concatenate into [NH, 4*DH, DH] at load time
            let fgate_w = load_vec(&content, &mut reader, &p("fgate.weight"), &device)?;
            let igate_w = load_vec(&content, &mut reader, &p("igate.weight"), &device)?;
            let zgate_w = load_vec(&content, &mut reader, &p("zgate.weight"), &device)?;
            let ogate_w = load_vec(&content, &mut reader, &p("ogate.weight"), &device)?;
            let mut fizo_w = vec![0.0f32; nh * 4 * wpp];
            for h in 0..nh {
                fizo_w[h*4*wpp..       h*4*wpp+wpp  ].copy_from_slice(&fgate_w[h*wpp..(h+1)*wpp]);
                fizo_w[h*4*wpp+wpp..   h*4*wpp+2*wpp].copy_from_slice(&igate_w[h*wpp..(h+1)*wpp]);
                fizo_w[h*4*wpp+2*wpp.. h*4*wpp+3*wpp].copy_from_slice(&zgate_w[h*wpp..(h+1)*wpp]);
                fizo_w[h*4*wpp+3*wpp.. h*4*wpp+4*wpp].copy_from_slice(&ogate_w[h*wpp..(h+1)*wpp]);
            }

            let raw_kernel = load_vec(&content, &mut reader, &p("slstm_kernel"), &device)?;
            // Transpose kernel [NH, DH, NG*DH] → [NH, NG*DH, DH] so the inner dot-product
            // dim (di) is contiguous, enabling simd_dot in compute_ry.
            let ng = 4usize;
            let mut slstm_kernel_t = vec![0.0f32; nh * ng * dh * dh];
            for head in 0..nh {
                for gate_d in 0..(ng * dh) {
                    for di in 0..dh {
                        slstm_kernel_t[head * ng * dh * dh + gate_d * dh + di] =
                            raw_kernel[head * dh * ng * dh + di * ng * dh + gate_d];
                    }
                }
            }

            blocks.push(Block {
                norm_slstm:    load_vec(&content, &mut reader, &p("norm_slstm"), &device)?,
                fizo_w,
                slstm_kernel_t,
                slstm_bias:    load_vec(&content, &mut reader, &p("slstm_bias"), &device)?,
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

            // Pre-allocate scratch buffers shared across all 12 sLSTM blocks.
            // Each block previously re-allocated these on every call; pre-allocating once
            // eliminates 12 × ~(1.5 MB) of heap churn per forecast() call.
            let ng = 4usize;
            let d  = cfg.embedding_dim;
            let mut sc_xg    = vec![0.0f32; num_patches * ng * d]; // fused gate output / x_g
            let mut sc_hout  = vec![0.0f32; num_patches * d];       // h_out
            let mut sc_y     = vec![0.0f32; num_patches * d];       // group-norm output y
            let mut sc_xn    = vec![0.0f32; num_patches * d];       // pre-norm copy x_n
            let mut sc_raw   = vec![0.0f32; ng * d];                // raw = wx + ry + bias
            let mut sc_ry_raw = vec![0.0f32; ng * d];               // compute_ry intermediate
            let mut sc_ry_out = vec![0.0f32; ng * d];               // compute_ry output
            let mut sc_hnew  = vec![0.0f32; d];                     // h_new (swapped, not re-alloc)
            let mut sc_cnew  = vec![0.0f32; d];
            let mut sc_nnew  = vec![0.0f32; d];
            let mut sc_mnew  = vec![0.0f32; d];

            // 12 sLSTM blocks
            for block in &self.blocks {
                hidden = self.forward_slstm_block(
                    &hidden, num_patches, block,
                    &mut sc_xg, &mut sc_hout, &mut sc_y, &mut sc_xn,
                    &mut sc_raw, &mut sc_ry_raw, &mut sc_ry_out,
                    &mut sc_hnew, &mut sc_cnew, &mut sc_nnew, &mut sc_mnew,
                )?;
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

    #[allow(clippy::too_many_arguments)]
    fn forward_slstm_block(
        &self, x: &[f32], s: usize, blk: &Block,
        sc_xg:    &mut [f32],  // [s * ng * d]  — fused gate projection output
        sc_hout:  &mut [f32],  // [s * d]        — sLSTM hidden outputs
        sc_y:     &mut [f32],  // [s * d]        — group-norm output
        sc_xn:    &mut [f32],  // [s * d]        — pre-norm scratch
        sc_raw:   &mut [f32],  // [ng * d]       — wx + ry + bias per step
        sc_ry_raw: &mut [f32], // [ng * d]       — compute_ry intermediate
        sc_ry_out: &mut [f32], // [ng * d]       — compute_ry output
        sc_hnew: &mut Vec<f32>, sc_cnew: &mut Vec<f32>,
        sc_nnew: &mut Vec<f32>, sc_mnew: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let d  = cfg.embedding_dim;
        let nh = cfg.num_heads;
        let dh = cfg.head_dim();
        let ng = 4usize;

        // Pre-norm: reuse sc_xn scratch to avoid allocation
        sc_xn[..s * d].copy_from_slice(&x[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_slstm, d, 1e-6);

        // Single fused gate projection: [S, D] → [S, ng*D] in one pass over x_n.
        // Replaces 4 separate headwise_linear_batch calls + interleaving copy loop.
        headwise_linear_batch_ng(&sc_xn[..s * d], &blk.fizo_w, s, nh, dh, ng, sc_xg);

        // Sequential sLSTM recurrence — state lives on the stack, swapped not re-allocated
        let mut h = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        let mut n = vec![0.0f32; d];
        let mut m = vec![f32::NEG_INFINITY; d];

        for t in 0..s {
            let wx = &sc_xg[t * ng * d..(t + 1) * ng * d];
            compute_ry(&h, &blk.slstm_kernel_t, nh, dh, ng, sc_ry_raw, sc_ry_out);

            // Reuse sc_raw to avoid per-step allocation of raw = wx + ry + bias
            for i in 0..ng * d {
                sc_raw[i] = wx[i] + sc_ry_out[i] + blk.slstm_bias[i];
            }

            // is_first ≡ t == 0: n starts as zeros and is never zero again after step 0
            let is_first = t == 0;

            // gate indices in flat [NG, NH, DH] layout:
            //   g=0 (offset 0) → iraw (from fgate module, input gate)
            //   g=1 (offset D) → fraw (from igate module, forget gate)
            //   g=2 (offset 2D) → zraw (cell gate)
            //   g=3 (offset 3D) → oraw (output gate)
            simd_gate_update(
                &sc_raw[..d], &sc_raw[d..2*d], &sc_raw[2*d..3*d], &sc_raw[3*d..],
                &c, &n, &m,
                sc_cnew, sc_nnew, sc_hnew, sc_mnew,
                is_first,
            );

            // Swap state vectors — zero allocation, just pointer swap
            std::mem::swap(&mut h, sc_hnew);
            std::mem::swap(&mut c, sc_cnew);
            std::mem::swap(&mut n, sc_nnew);
            std::mem::swap(&mut m, sc_mnew);
            sc_hout[t * d..(t + 1) * d].copy_from_slice(&h);
        }

        // MultiHeadLayerNorm (group norm per head per token) + reshape to [S, D]
        for t in 0..s {
            let h_t = &sc_hout[t * d..(t + 1) * d];
            for head in 0..nh {
                let start = head * dh;
                let h_slice = &h_t[start..start + dh];
                let mean = h_slice.iter().sum::<f32>() / dh as f32;
                let var = h_slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dh as f32;
                let inv_std = 1.0 / (var + 1e-5f32).sqrt();
                let w_slice = &blk.group_norm_w[start..start + dh];
                for d_i in 0..dh {
                    sc_y[t * d + start + d_i] =
                        (h_t[start + d_i] - mean) * inv_std * (1.0 + w_slice[d_i]);
                }
            }
        }

        // Residual: x + y
        let mut x_out = x.to_vec();
        for i in 0..s * d {
            x_out[i] += sc_y[i];
        }

        // FFN pre-norm (reuse sc_xn)
        sc_xn[..s * d].copy_from_slice(&x_out[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_ffn, d, 1e-6);

        // FFN: SiLU gated: (silu(gate(x)) * up(x)) → down
        let ffn_out = ffn_forward(&sc_xn[..s * d], s, d, cfg.ffn_up_dim, blk, &self.device)?;

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

// ---------------------------------------------------------------------------
// SIMD kernels (cross-platform: SSE2 / AVX2 / AVX-512 / NEON / WASM / scalar)
// ---------------------------------------------------------------------------

simd_runtime_generate!(
    fn simd_sq_sum(row: &[f32]) -> f32 {
        let mut r = &row[..];
        let mut acc = S::Vf32::zeroes();
        while r.len() >= S::Vf32::WIDTH {
            let v = S::Vf32::load_from_slice(r);
            acc = v.mul_add(v, acc);
            r = &r[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for &x in r { sum += x * x; }
        sum
    }
);

simd_runtime_generate!(
    fn simd_scale_weight(row: &mut [f32], w: &[f32], scale: f32) {
        let sc = S::Vf32::set1(scale);
        let mut r = &mut row[..];
        let mut wv = &w[..];
        while r.len() >= S::Vf32::WIDTH {
            let v = S::Vf32::load_from_slice(r);
            let wi = S::Vf32::load_from_slice(wv);
            (v * sc * wi).copy_to_slice(r);
            r = &mut r[S::Vf32::WIDTH..];
            wv = &wv[S::Vf32::WIDTH..];
        }
        for i in 0..r.len() { r[i] *= scale * wv[i]; }
    }
);

simd_runtime_generate!(
    fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut aa = &a[..];
        let mut bb = &b[..];
        let mut acc = S::Vf32::zeroes();
        while aa.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(aa);
            let vb = S::Vf32::load_from_slice(bb);
            acc = va.mul_add(vb, acc);
            aa = &aa[S::Vf32::WIDTH..];
            bb = &bb[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for (&x, &y) in aa.iter().zip(bb.iter()) { sum += x * y; }
        sum
    }
);

simd_runtime_generate!(
    fn simd_gate_update(
        iraw: &[f32], fraw: &[f32], zraw: &[f32], oraw: &[f32],
        c: &[f32], n: &[f32], m: &[f32],
        cnew: &mut [f32], nnew: &mut [f32], hnew: &mut [f32], mnew: &mut [f32],
        is_first: bool,
    ) {
        let clamp = S::Vf32::set1(15.0f32);
        let one   = S::Vf32::set1(1.0f32);
        let eps   = S::Vf32::set1(1e-8f32);
        let zero  = S::Vf32::zeroes();

        let mut ir = &iraw[..]; let mut fr = &fraw[..];
        let mut zr = &zraw[..]; let mut or_ = &oraw[..];
        let mut cv = &c[..];    let mut nv = &n[..];    let mut mv = &m[..];
        let mut cnw = &mut cnew[..]; let mut nnw = &mut nnew[..];
        let mut hnw = &mut hnew[..]; let mut mnw = &mut mnew[..];

        while ir.len() >= S::Vf32::WIDTH {
            let iv  = S::Vf32::load_from_slice(ir);
            let fv  = S::Vf32::load_from_slice(fr).min(clamp);
            let zv  = S::Vf32::load_from_slice(zr);
            let ov  = S::Vf32::load_from_slice(or_);
            let cv_ = S::Vf32::load_from_slice(cv);
            let nv_ = S::Vf32::load_from_slice(nv);
            let mv_ = S::Vf32::load_from_slice(mv);

            // log_sigmoid(fv): stable two-branch form, select by sign
            let ls_pos = -(one + (-fv).exp_u35()).ln_u35();     // fv >= 0: -ln(1+exp(-fv))
            let ls_neg = fv - (one + fv.exp_u35()).ln_u35();     // fv < 0:  fv - ln(1+exp(fv))
            let log_sig_f = fv.cmp_lt(zero).blendv(ls_pos, ls_neg);

            let logfplusm = mv_ + log_sig_f;
            let mnew_v = if is_first { iv } else { iv.max(logfplusm) };

            let ogate = one / (one + (-ov).exp_u35());            // sigmoid(ov)
            let igate = (iv - mnew_v).exp_u35().min(one);
            let fgate = (logfplusm - mnew_v).exp_u35().min(one);
            let zgate = zv.tanh_u35();

            let cnew_v = fgate.mul_add(cv_, igate * zgate);
            let nnew_v = fgate.mul_add(nv_, igate);

            let mask  = nnew_v.abs().cmp_gt(eps);
            let hnew_v = mask.blendv(zero, ogate * cnew_v / nnew_v);

            cnew_v.copy_to_slice(cnw); nnew_v.copy_to_slice(nnw);
            hnew_v.copy_to_slice(hnw); mnew_v.copy_to_slice(mnw);

            ir  = &ir[S::Vf32::WIDTH..];  fr  = &fr[S::Vf32::WIDTH..];
            zr  = &zr[S::Vf32::WIDTH..];  or_ = &or_[S::Vf32::WIDTH..];
            cv  = &cv[S::Vf32::WIDTH..];  nv  = &nv[S::Vf32::WIDTH..];
            mv  = &mv[S::Vf32::WIDTH..];
            cnw = &mut cnw[S::Vf32::WIDTH..]; nnw = &mut nnw[S::Vf32::WIDTH..];
            hnw = &mut hnw[S::Vf32::WIDTH..]; mnw = &mut mnw[S::Vf32::WIDTH..];
        }

        for i in 0..ir.len() {
            let iv_s = ir[i]; let fv_s = fr[i].min(15.0); let zv_s = zr[i]; let ov_s = or_[i];
            let c_s = cv[i]; let n_s = nv[i]; let m_s = mv[i];
            let ls = if fv_s >= 0.0 { -(1.0 + (-fv_s).exp()).ln() } else { fv_s - (1.0 + fv_s.exp()).ln() };
            let lfpm = m_s + ls;
            let mn = if is_first { iv_s } else { iv_s.max(lfpm) };
            let og = 1.0 / (1.0 + (-ov_s).exp());
            let ig = (iv_s - mn).exp().min(1.0);
            let fg = (lfpm - mn).exp().min(1.0);
            let zg = zv_s.tanh();
            cnw[i] = fg * c_s + ig * zg;
            nnw[i] = fg * n_s + ig;
            hnw[i] = if nnw[i].abs() > 1e-8 { og * cnw[i] / nnw[i] } else { 0.0 };
            mnw[i] = mn;
        }
    }
);

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
        let ss = simd_sq_sum(row);
        let scale = 1.0 / (ss / d as f32 + eps).sqrt();
        simd_scale_weight(row, w, scale);
    }
}

/// Fused headwise linear for ng gate groups: [S, D] → [S, ng*D] written into `out`.
///
/// w layout: [NH, ng*DH, DH] — for head h, gate g: rows g*DH..(g+1)*DH hold w[h,g,:].
/// Output layout: out[t, g, h, o] at flat index t*ng*d + g*d + h*dh + o.
/// This matches the x_g layout expected by the sLSTM recurrence, so no interleaving
/// copy is needed after calling this function.
fn headwise_linear_batch_ng(x: &[f32], w: &[f32], s: usize, nh: usize, dh: usize, ng: usize, out: &mut [f32]) {
    let d    = nh * dh;
    let ngdh = ng * dh;
    for t in 0..s {
        for h in 0..nh {
            let x_h     = &x[t * d + h * dh..t * d + (h + 1) * dh];
            let w_h_off = h * ngdh * dh;
            for g in 0..ng {
                let out_base    = t * ng * d + g * d + h * dh;
                let w_gate_off  = w_h_off + g * dh * dh;
                for o in 0..dh {
                    let w_row = &w[w_gate_off + o * dh..w_gate_off + (o + 1) * dh];
                    out[out_base + o] = simd_dot(x_h, w_row);
                }
            }
        }
    }
}

/// Compute recurrent contribution Ry from h_prev and the sLSTM kernel.
///
/// kernel_t layout: [NH, NG*DH, DH] (transposed at load time from [NH, DH, NG*DH])
/// so the inner dot-product dimension (di) is contiguous, enabling simd_dot.
/// Output written into `out`: [NG*NH*DH = 2048] in [NG, NH, DH] order.
fn compute_ry(
    h: &[f32], kernel_t: &[f32],
    nh: usize, dh: usize, ng: usize,
    ry_raw: &mut [f32],
    out: &mut [f32],
) {
    // ry_raw[NH, NG*DH]: for each head, dot h_head with each row of kernel_t[head]
    for head in 0..nh {
        let h_h    = &h[head * dh..(head + 1) * dh];
        let k_head = &kernel_t[head * ng * dh * dh..];
        for gate_d in 0..ng * dh {
            let k_row = &k_head[gate_d * dh..(gate_d + 1) * dh]; // contiguous — simd_dot applies
            ry_raw[head * ng * dh + gate_d] = simd_dot(h_h, k_row);
        }
    }
    // Permute [NH, NG, DH] → [NG, NH, DH]
    for head in 0..nh {
        for g in 0..ng {
            for d in 0..dh {
                out[g * nh * dh + head * dh + d] = ry_raw[head * ng * dh + g * dh + d];
            }
        }
    }
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
