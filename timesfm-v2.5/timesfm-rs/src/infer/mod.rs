mod rope;

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use rope::RopeCache;

const NORM_EPS: f64 = 1e-6;
const REVIN_TOL: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Architecture constants (TimesFM 2.5 200M)
// ---------------------------------------------------------------------------
const D_MODEL: usize = 1280;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 80;
const N_LAYERS: usize = 20;
const INPUT_PATCH: usize = 32;
const OUTPUT_PATCH: usize = 128;
const N_OUTPUTS: usize = 10;
const DECODE_IDX: usize = 5;
const M_PATCHES: usize = OUTPUT_PATCH / INPUT_PATCH; // 4
const ROPE_THETA: f64 = 10000.0;
const MAX_SEQ: usize = 16384 / INPUT_PATCH + 256; // generous upper bound

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct ResidualBlockW {
    hidden_w: Tensor,
    hidden_b: Option<Tensor>,
    output_w: Tensor,
    output_b: Option<Tensor>,
    skip_w: Tensor,
    skip_b: Option<Tensor>,
}

struct AttnW {
    qkv_w: Tensor,        // [3*D_MODEL, D_MODEL]
    out_w: Tensor,        // [D_MODEL, D_MODEL]
    q_norm_w: Tensor,     // [HEAD_DIM]
    k_norm_w: Tensor,     // [HEAD_DIM]
    q_scale: Tensor,      // [HEAD_DIM] cached as Tensor to avoid alloc per attention forward
    pre_norm_w: Tensor,   // [D_MODEL]
    post_norm_w: Tensor,  // [D_MODEL]
}

struct FfnW {
    up_w: Tensor,         // [D_MODEL, D_MODEL]
    down_w: Tensor,       // [D_MODEL, D_MODEL]
    pre_norm_w: Tensor,   // [D_MODEL]
    post_norm_w: Tensor,  // [D_MODEL]
}

struct BlockW {
    attn: AttnW,
    ffn: FfnW,
}

pub struct TimesFMModel {
    device: Device,
    rope: RopeCache,
    tokenizer: ResidualBlockW,
    blocks: Vec<BlockW>,
    out_point: ResidualBlockW,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
}

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

fn load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    let qt = content
        .tensor(reader, name, device)
        .with_context(|| format!("load tensor '{name}'"))?;
    Ok(qt.dequantize(device)?.to_dtype(DType::F32)?)
}

/// Load a weight stored as (d_out, d_in) in PyTorch, reversed in GGUF.
/// Candle returns the GGUF shape [d_in, d_out]; we detect and transpose.
fn load_weight(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    expected_d_out: usize,
    device: &Device,
) -> Result<Tensor> {
    let w = load_tensor(content, reader, name, device)?;
    if w.dim(0)? != expected_d_out {
        Ok(w.t()?.contiguous()?)
    } else {
        Ok(w)
    }
}

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    hidden_d_out: usize,
    out_d_out: usize,
    with_bias: bool,
    device: &Device,
) -> Result<ResidualBlockW> {
    let hidden_w = load_weight(content, reader, &format!("{prefix}.hidden.weight"), hidden_d_out, device)?;
    let hidden_b = if with_bias {
        Some(load_tensor(content, reader, &format!("{prefix}.hidden.bias"), device)?)
    } else {
        None
    };
    let output_w = load_weight(content, reader, &format!("{prefix}.output.weight"), out_d_out, device)?;
    let output_b = if with_bias {
        Some(load_tensor(content, reader, &format!("{prefix}.output.bias"), device)?)
    } else {
        None
    };
    let skip_w = load_weight(content, reader, &format!("{prefix}.skip.weight"), out_d_out, device)?;
    let skip_b = if with_bias {
        Some(load_tensor(content, reader, &format!("{prefix}.skip.bias"), device)?)
    } else {
        None
    };
    Ok(ResidualBlockW { hidden_w, hidden_b, output_w, output_b, skip_w, skip_b })
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() }
}

fn compute_q_scale(raw: Vec<f32>) -> Vec<f32> {
    let factor = 1.442695041f32 / (HEAD_DIM as f32).sqrt();
    raw.into_iter().map(|x| factor * softplus(x)).collect()
}

impl TimesFMModel {
    pub fn load(gguf_path: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        // Tokenizer ResidualBlock: in=64, hidden=1280, out=1280, bias=true
        let tokenizer = load_residual_block(
            &content, &mut reader, "tokenizer", D_MODEL, D_MODEL, true, &device,
        )?;

        // 20 transformer blocks
        let mut blocks = Vec::with_capacity(N_LAYERS);
        for n in 0..N_LAYERS {
            let b = format!("blk.{n}");
            let qkv_raw = load_weight(&content, &mut reader, &format!("{b}.attn_qkv.weight"), 3 * D_MODEL, &device)?;
            let out_w = load_weight(&content, &mut reader, &format!("{b}.attn_out.weight"), D_MODEL, &device)?;
            let q_norm_w = load_tensor(&content, &mut reader, &format!("{b}.attn_q_norm.weight"), &device)?;
            let k_norm_w = load_tensor(&content, &mut reader, &format!("{b}.attn_k_norm.weight"), &device)?;
            let q_scale_raw = load_tensor(&content, &mut reader, &format!("{b}.attn_q_scale.weight"), &device)?
                .to_vec1::<f32>()?;
            let q_scale = Tensor::from_vec(compute_q_scale(q_scale_raw), (HEAD_DIM,), &device)?;
            let pre_norm_w  = load_tensor(&content, &mut reader, &format!("{b}.pre_attn_norm.weight"), &device)?;
            let post_norm_w = load_tensor(&content, &mut reader, &format!("{b}.post_attn_norm.weight"), &device)?;

            let up_w   = load_weight(&content, &mut reader, &format!("{b}.ffn_up.weight"),   D_MODEL, &device)?;
            let down_w = load_weight(&content, &mut reader, &format!("{b}.ffn_down.weight"), D_MODEL, &device)?;
            let pre_ff_norm_w  = load_tensor(&content, &mut reader, &format!("{b}.pre_ff_norm.weight"),  &device)?;
            let post_ff_norm_w = load_tensor(&content, &mut reader, &format!("{b}.post_ff_norm.weight"), &device)?;

            blocks.push(BlockW {
                attn: AttnW {
                    qkv_w: qkv_raw,
                    out_w,
                    q_norm_w,
                    k_norm_w,
                    q_scale,
                    pre_norm_w,
                    post_norm_w,
                },
                ffn: FfnW {
                    up_w,
                    down_w,
                    pre_norm_w: pre_ff_norm_w,
                    post_norm_w: post_ff_norm_w,
                },
            });
        }

        // Output projection (no bias, hidden=1280, out=1280)
        let out_point = load_residual_block(
            &content, &mut reader, "out_point", D_MODEL, D_MODEL, false, &device,
        )?;

        let rope = RopeCache::new(HEAD_DIM, MAX_SEQ, ROPE_THETA, &device)?;

        Ok(Self { device, rope, tokenizer, blocks, out_point, causal_mask_cache: Mutex::new(HashMap::new()) })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Forecast a univariate time series.
    ///
    /// Returns `[N_OUTPUTS][prediction_length]` where index 0 is the point
    /// forecast and indices 1-9 are quantile forecasts (q0.1 … q0.9).
    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<Vec<Vec<f32>>> {
        let p = INPUT_PATCH;
        let o = OUTPUT_PATCH;
        let q = N_OUTPUTS;
        let m = M_PATCHES;

        // Pad front so len is divisible by p
        let len_front = if context.len() % p == 0 { 0 } else { p - context.len() % p };
        let mut vals = vec![0.0f32; len_front + context.len()];
        vals[len_front..].copy_from_slice(context);
        let mut mask = vec![true; len_front]; // True = masked (padding)
        mask.extend(vec![false; context.len()]);

        let n_ctx = (len_front + context.len()) / p;

        // Compute cumulative running stats per patch
        let mut patch_mus = Vec::with_capacity(n_ctx);
        let mut patch_sigmas = Vec::with_capacity(n_ctx);
        let (mut rs_n, mut rs_mu, mut rs_sigma) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n_ctx {
            let pv = &vals[i * p..(i + 1) * p];
            let pm = &mask[i * p..(i + 1) * p];
            (rs_n, rs_mu, rs_sigma) = update_running_stats(rs_n, rs_mu, rs_sigma, pv, pm);
            patch_mus.push(rs_mu);
            patch_sigmas.push(rs_sigma);
        }
        let (last_n, last_mu, last_sigma) = (rs_n, rs_mu, rs_sigma);

        // Build normalized tokenizer input for context patches
        let ctx_input = build_tokenizer_input(&vals, &mask, n_ctx, p, &patch_mus, &patch_sigmas);

        // Prefill: full forward pass collecting per-layer KV cache
        let num_decode_steps = if prediction_length <= o { 0 } else { (prediction_length - 1) / o };
        let total_steps = 1 + num_decode_steps;
        let mut all_outputs: Vec<Vec<[f32; N_OUTPUTS]>> = Vec::with_capacity(total_steps);

        let (ctx_out, mut kv_cache) = self.prefill(&ctx_input, n_ctx)?;
        let ctx_denorm = denorm_flat(&ctx_out, &patch_mus, &patch_sigmas, o, q)?;
        all_outputs.push(ctx_denorm[n_ctx - 1].clone());

        // AR decode: process M_PATCHES=4 new patches per step with cached K/V
        let (mut ar_n, mut ar_mu, mut ar_sigma) = (last_n, last_mu, last_sigma);
        let mut last_ar: Vec<f32> = ctx_denorm[n_ctx - 1].iter().map(|row| row[DECODE_IDX]).collect();

        for step in 0..num_decode_steps {
            let new_vals_flat = last_ar.clone();
            let new_mask_flat = vec![false; o];

            let mut new_mus = Vec::with_capacity(m);
            let mut new_sigmas = Vec::with_capacity(m);
            for mi in 0..m {
                let pv = &new_vals_flat[mi * p..(mi + 1) * p];
                let pm = &new_mask_flat[mi * p..(mi + 1) * p];
                (ar_n, ar_mu, ar_sigma) = update_running_stats(ar_n, ar_mu, ar_sigma, pv, pm);
                new_mus.push(ar_mu);
                new_sigmas.push(ar_sigma);
            }

            let new_input = build_tokenizer_input(
                &new_vals_flat, &new_mask_flat, m, p, &new_mus, &new_sigmas,
            );
            let rope_offset = n_ctx + m * step;
            let new_out = self.decode_chunk(&new_input, &mut kv_cache, rope_offset)?;
            let last_m_denorm = denorm_flat(&new_out, &new_mus, &new_sigmas, o, q)?;

            let step_out = &last_m_denorm[m - 1];
            last_ar = step_out.iter().map(|row| row[DECODE_IDX]).collect();
            all_outputs.push(step_out.clone());
        }

        // Assemble [N_OUTPUTS][prediction_length]
        let total_available = all_outputs.len() * o;
        let n_take = prediction_length.min(total_available);
        let mut result: Vec<Vec<f32>> = vec![Vec::with_capacity(n_take); q];
        'outer: for step in &all_outputs {
            for timestep in step {
                for qi in 0..q {
                    if result[qi].len() >= prediction_length { break 'outer; }
                    result[qi].push(timestep[qi]);
                }
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Prefill: full forward + collect per-layer KV cache
    // KV tensors stored as [1, N_HEADS, n_patches, HEAD_DIM]
    // -----------------------------------------------------------------------

    fn prefill(
        &self,
        tokenizer_input: &Tensor,  // [n_patches, 2*INPUT_PATCH]
        n_patches: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let x = forward_residual_block(tokenizer_input, &self.tokenizer, true)?;
        let mut hidden = x.unsqueeze(0)?;
        let causal = {
            let mut cache = self.causal_mask_cache.lock().unwrap();
            if !cache.contains_key(&n_patches) {
                cache.insert(n_patches, make_causal_mask(n_patches, 0, &self.device)?);
            }
            cache[&n_patches].clone()
        };
        let mut kv_cache: Vec<(Tensor, Tensor)> = Vec::with_capacity(N_LAYERS);
        for block in &self.blocks {
            let (h_out, k, v) = self.prefill_block(hidden, block, n_patches, &causal)?;
            hidden = h_out;
            kv_cache.push((k, v));
        }
        let out_seq = hidden.squeeze(0)?;
        let out = forward_residual_block(&out_seq, &self.out_point, false)?;
        Ok((out, kv_cache))
    }

    fn prefill_block(
        &self,
        x: Tensor,
        w: &BlockW,
        n_patches: usize,
        causal: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = rms_norm(&x, &w.attn.pre_norm_w, NORM_EPS)?;
        let (attn_out, k, v) = self.prefill_attn(normed, &w.attn, n_patches, causal)?;
        let attn_out = (rms_norm(&attn_out, &w.attn.post_norm_w, NORM_EPS)? + &x)?;
        let normed_ff = rms_norm(&attn_out, &w.ffn.pre_norm_w, NORM_EPS)?;
        let ff_out = self.forward_ffn(normed_ff, &w.ffn)?;
        let out = (rms_norm(&ff_out, &w.ffn.post_norm_w, NORM_EPS)? + &attn_out)?;
        Ok((out, k, v))
    }

    fn prefill_attn(
        &self,
        x: Tensor,       // [1, n_patches, D_MODEL] pre-normed
        w: &AttnW,
        n_patches: usize,
        causal: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {  // (output, K, V) K/V: [1,N_HEADS,n,HEAD_DIM]
        let qkv = linear(&x, &w.qkv_w, None)?;
        let q = qkv.narrow(D::Minus1, 0, D_MODEL)?;
        let k = qkv.narrow(D::Minus1, D_MODEL, D_MODEL)?;
        let v = qkv.narrow(D::Minus1, 2 * D_MODEL, D_MODEL)?;

        let q = q.reshape((1, n_patches, N_HEADS, HEAD_DIM))?;
        let k = k.reshape((1, n_patches, N_HEADS, HEAD_DIM))?;
        let v = v.reshape((1, n_patches, N_HEADS, HEAD_DIM))?;

        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;
        let q = rms_norm(&q, &w.q_norm_w, NORM_EPS)?;
        let k = rms_norm(&k, &w.k_norm_w, NORM_EPS)?;

        let q = q.broadcast_mul(&w.q_scale)?;

        let q = q.permute([0, 2, 1, 3])?.contiguous()?;   // [1, N_HEADS, n, HEAD_DIM]
        let k = k.permute([0, 2, 1, 3])?.contiguous()?;
        let v = v.permute([0, 2, 1, 3])?.contiguous()?;

        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        let scores = scores.broadcast_add(causal)?;
        let attn_w = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn_w.matmul(&v)?;
        let ctx = ctx.permute([0, 2, 1, 3])?.contiguous()?.reshape((1, n_patches, D_MODEL))?;
        Ok((linear(&ctx, &w.out_w, None)?, k, v))
    }

    // -----------------------------------------------------------------------
    // KV-cached decode: process M_PATCHES new patches, append to cache
    // -----------------------------------------------------------------------

    fn decode_chunk(
        &self,
        new_input: &Tensor,                    // [M_PATCHES, 2*INPUT_PATCH]
        kv_cache: &mut Vec<(Tensor, Tensor)>,  // mutated: K/V grow by M_PATCHES per call
        rope_offset: usize,                    // = n_ctx + M_PATCHES * step
    ) -> Result<Tensor> {                      // → [M_PATCHES, D_MODEL]
        let x = forward_residual_block(new_input, &self.tokenizer, true)?;
        let mut hidden = x.unsqueeze(0)?;
        let cached_len = rope_offset;
        let decode_mask = make_decode_mask(M_PATCHES, cached_len, &self.device)?;
        for (li, block) in self.blocks.iter().enumerate() {
            // decode_block_kv returns the extended K/V (old cache + new M_PATCHES).
            // Store directly — no second Tensor::cat needed.
            let (h_out, k_new, v_new) =
                self.decode_block_kv(hidden, block, &kv_cache[li], rope_offset, &decode_mask)?;
            hidden = h_out;
            kv_cache[li] = (k_new, v_new);
        }
        let out_seq = hidden.squeeze(0)?;
        forward_residual_block(&out_seq, &self.out_point, false)
    }

    fn decode_block_kv(
        &self,
        x: Tensor,
        w: &BlockW,
        cache: &(Tensor, Tensor),
        rope_offset: usize,
        decode_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = rms_norm(&x, &w.attn.pre_norm_w, NORM_EPS)?;
        let (attn_out, k_new, v_new) =
            self.decode_attn_kv(normed, &w.attn, cache, rope_offset, decode_mask)?;
        let attn_out = (rms_norm(&attn_out, &w.attn.post_norm_w, NORM_EPS)? + &x)?;
        let normed_ff = rms_norm(&attn_out, &w.ffn.pre_norm_w, NORM_EPS)?;
        let ff_out = self.forward_ffn(normed_ff, &w.ffn)?;
        let out = (rms_norm(&ff_out, &w.ffn.post_norm_w, NORM_EPS)? + &attn_out)?;
        Ok((out, k_new, v_new))
    }

    fn decode_attn_kv(
        &self,
        x: Tensor,                  // [1, M_PATCHES, D_MODEL] pre-normed
        w: &AttnW,
        cache: &(Tensor, Tensor),   // ([1,N_HEADS,cached,HEAD_DIM], same)
        rope_offset: usize,
        mask: &Tensor,              // [1,1,M_PATCHES,cached+M_PATCHES]
    ) -> Result<(Tensor, Tensor, Tensor)> {  // (output, K_new, V_new)
        let qkv = linear(&x, &w.qkv_w, None)?;
        let q = qkv.narrow(D::Minus1, 0, D_MODEL)?;
        let k = qkv.narrow(D::Minus1, D_MODEL, D_MODEL)?;
        let v = qkv.narrow(D::Minus1, 2 * D_MODEL, D_MODEL)?;

        let q = q.reshape((1, M_PATCHES, N_HEADS, HEAD_DIM))?;
        let k = k.reshape((1, M_PATCHES, N_HEADS, HEAD_DIM))?;
        let v = v.reshape((1, M_PATCHES, N_HEADS, HEAD_DIM))?;

        let q = self.rope.apply(&q, rope_offset)?;
        let k = self.rope.apply(&k, rope_offset)?;
        let q = rms_norm(&q, &w.q_norm_w, NORM_EPS)?;
        let k = rms_norm(&k, &w.k_norm_w, NORM_EPS)?;

        let q = q.broadcast_mul(&w.q_scale)?;

        let q = q.permute([0, 2, 1, 3])?.contiguous()?;   // [1, N_HEADS, M, HEAD_DIM]
        let k = k.permute([0, 2, 1, 3])?.contiguous()?;
        let v = v.permute([0, 2, 1, 3])?.contiguous()?;

        // Extend cache: [1, N_HEADS, cached+M, HEAD_DIM].
        // Return k_full/v_full so the caller can store them directly without a second cat.
        let k_full = Tensor::cat(&[&cache.0, &k], 2)?;
        let v_full = Tensor::cat(&[&cache.1, &v], 2)?;

        let scores = q.matmul(&k_full.transpose(D::Minus1, D::Minus2)?)?;
        let scores = scores.broadcast_add(mask)?;
        let attn_w = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn_w.matmul(&v_full)?;
        let ctx = ctx.permute([0, 2, 1, 3])?.contiguous()?.reshape((1, M_PATCHES, D_MODEL))?;
        Ok((linear(&ctx, &w.out_w, None)?, k_full, v_full))
    }

    fn forward_ffn(&self, x: Tensor, w: &FfnW) -> Result<Tensor> {
        let h = linear(&x, &w.up_w, None)?;
        let h = candle_nn::ops::silu(&h)?;
        linear(&h, &w.down_w, None)
    }
}

// ---------------------------------------------------------------------------
// ResidualBlock forward (tokenizer has bias, output projections do not)
// ---------------------------------------------------------------------------

fn forward_residual_block(x: &Tensor, w: &ResidualBlockW, _has_bias: bool) -> Result<Tensor> {
    let h = linear(x, &w.hidden_w, w.hidden_b.as_ref())?;
    let h = candle_nn::ops::silu(&h)?;
    let out = linear(&h, &w.output_w, w.output_b.as_ref())?;
    let skip = linear(x, &w.skip_w, w.skip_b.as_ref())?;
    Ok((out + skip)?)
}

// ---------------------------------------------------------------------------
// Primitive ops
// ---------------------------------------------------------------------------

fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let shape = x.dims().to_vec();
    let d_in = *shape.last().unwrap();
    let batch: usize = shape[..shape.len() - 1].iter().product();
    let d_out = w.dim(0)?;
    let x_flat = x.reshape((batch, d_in))?;
    let out_flat = x_flat.matmul(&w.t()?)?;
    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(d_out);
    let out = out_flat.reshape(out_shape)?;
    if let Some(bias) = b { Ok(out.broadcast_add(bias)?) } else { Ok(out) }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (var + eps)?.sqrt()?;
    Ok(x.broadcast_div(&rms)?.broadcast_mul(weight)?)
}

/// Additive causal mask: 0 where q>=k, -inf where q<k. Shape [1,1,seq,seq].
fn make_causal_mask(seq: usize, num_masked: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..seq).flat_map(|q| {
        (0..seq).map(move |k| {
            if k <= q && k >= num_masked { 0.0f32 } else { f32::NEG_INFINITY }
        })
    }).collect();
    Ok(Tensor::from_vec(data, (seq, seq), device)?
        .unsqueeze(0)?.unsqueeze(0)?)
}

/// Decode-step causal mask: shape [1,1,new_len,cache_len+new_len].
/// New query i can attend to all cached tokens plus new tokens 0..=i.
fn make_decode_mask(new_len: usize, cache_len: usize, device: &Device) -> Result<Tensor> {
    let total = cache_len + new_len;
    let data: Vec<f32> = (0..new_len).flat_map(|q_rel| {
        (0..total).map(move |k_abs| {
            if k_abs <= cache_len + q_rel { 0.0f32 } else { f32::NEG_INFINITY }
        })
    }).collect();
    Ok(Tensor::from_vec(data, (new_len, total), device)?
        .unsqueeze(0)?.unsqueeze(0)?)
}

// ---------------------------------------------------------------------------
// Running statistics (mirrors Python update_running_stats)
// ---------------------------------------------------------------------------

fn update_running_stats(n: f32, mu: f32, sigma: f32, vals: &[f32], mask: &[bool]) -> (f32, f32, f32) {
    let mut inc_n = 0.0f32;
    let mut sum_x = 0.0f32;
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            inc_n += 1.0;
            sum_x += vals[i];
        }
    }
    let inc_mu = if inc_n == 0.0 { 0.0 } else { sum_x / inc_n };
    let inc_var = if inc_n == 0.0 { 0.0 } else {
        mask.iter().enumerate()
            .filter(|(_, &m)| !m)
            .map(|(i, _)| (vals[i] - inc_mu).powi(2))
            .sum::<f32>() / inc_n
    };
    let inc_sigma = inc_var.sqrt();

    let new_n = n + inc_n;
    let safe_new_n = if new_n == 0.0 { 1.0 } else { new_n };
    let new_mu = if new_n == 0.0 { 0.0 } else { (n * mu + inc_mu * inc_n) / safe_new_n };
    let t1 = n * sigma.powi(2);
    let t2 = inc_n * inc_sigma.powi(2);
    let t3 = n * (mu - new_mu).powi(2);
    let t4 = inc_n * (inc_mu - new_mu).powi(2);
    let new_var = if new_n == 0.0 { 0.0 } else { (t1 + t2 + t3 + t4) / safe_new_n };
    (new_n, new_mu, new_var.max(0.0).sqrt())
}

// ---------------------------------------------------------------------------
// Input preparation
// ---------------------------------------------------------------------------

/// Build tokenizer input tensor [n_patches, 2*INPUT_PATCH=64].
///
/// For each patch: cat([normed_values, mask_as_float], dim=-1).
/// Masked (padded) positions → value=0.0, mask=1.0.
fn build_tokenizer_input(
    vals: &[f32],
    mask: &[bool],
    n_patches: usize,
    p: usize,
    mus: &[f32],
    sigmas: &[f32],
) -> Tensor {
    let mut data = vec![0.0f32; n_patches * 2 * p];
    for pi in 0..n_patches {
        let mu = mus[pi];
        let sigma = sigmas[pi];
        let sigma_safe = if sigma < REVIN_TOL { 1.0f32 } else { sigma };
        for i in 0..p {
            let idx = pi * p + i;
            let is_masked = mask[idx];
            let normed = if is_masked { 0.0 } else { (vals[idx] - mu) / sigma_safe };
            let base = pi * 2 * p;
            data[base + i] = normed;                // value channel
            data[base + p + i] = if is_masked { 1.0 } else { 0.0 }; // mask channel
        }
    }
    Tensor::from_vec(data, (n_patches, 2 * p), &Device::Cpu).expect("build_tokenizer_input")
}

// ---------------------------------------------------------------------------
// Output denormalization
// ---------------------------------------------------------------------------

/// Denorm output [n_patches, D_MODEL] → flat vec of [n_patches][O][Q] triples.
///
/// Each patch i is denormed: out * sigma[i] + mu[i].
/// Then reshaped to [OUTPUT_PATCH, N_OUTPUTS].
fn denorm_flat(
    output: &Tensor,          // [n_patches, D_MODEL]
    mus: &[f32],
    sigmas: &[f32],
    o: usize,
    q: usize,
) -> Result<Vec<Vec<[f32; N_OUTPUTS]>>> {
    let flat = output.flatten_all()?.to_vec1::<f32>()?;
    let n = mus.len();
    let mut result: Vec<Vec<[f32; N_OUTPUTS]>> = vec![vec![[0.0f32; N_OUTPUTS]; o]; n];
    for pi in 0..n {
        let mu = mus[pi];
        let sigma = sigmas[pi];
        for ti in 0..o {
            for qi in 0..q {
                let raw = flat[pi * (o * q) + ti * q + qi];
                result[pi][ti][qi] = raw * sigma + mu;
            }
        }
    }
    Ok(result)
}
