//! Moirai-2.0-R-small inference engine.
//!
//! Architecture: causal transformer decoder with ResidualBlock in/out projections,
//! partial interleaved RoPE (first 32 of 64 head dims), BinaryAttentionBias,
//! PackedStdScaler normalization, and multi-token (4 patches/token) decoding loop.

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::config::Moirai2Config;

mod rope;
use rope::apply_partial_rope;

// ---------------------------------------------------------------------------
// Weight containers
// ---------------------------------------------------------------------------

struct ResidualBlockW {
    hidden_w:   Tensor, // Python [hidden_dim, in_dim]
    hidden_b:   Tensor, // [hidden_dim]
    output_w:   Tensor, // Python [out_dim, hidden_dim]
    output_b:   Tensor, // [out_dim]
    residual_w: Tensor, // Python [out_dim, in_dim]
    residual_b: Tensor, // [out_dim]
}

struct EncoderBlock {
    norm1_w:         Tensor, // [d_model]
    norm2_w:         Tensor,
    attn_qkv_w:      Tensor, // fused [3*d_model, d_model]
    attn_o_w:        Tensor,
    attn_qn_w:       Tensor, // [head_dim]  per-head QK norm (shared across heads)
    attn_kn_w:       Tensor,
    attn_vbias_t: Tensor,      // [n_heads, 1, 1] same-variate bias, precomputed at load
    ffn_fc1_w:       Tensor, // [d_ff, d_model]
    ffn_fc2_w:       Tensor, // [d_model, d_ff]
    ffn_gate_w:      Tensor, // [d_ff, d_model]
}

pub struct Moirai2Model {
    device:   Device,
    config:   Moirai2Config,
    in_proj:  ResidualBlockW,
    blocks:   Vec<EncoderBlock>,
    norm_f_w: Tensor,
    out_proj: ResidualBlockW,
    rope_cos: Vec<f32>,  // [max_pos * half_rope] precomputed cosines
    rope_sin: Vec<f32>,  // [max_pos * half_rope] precomputed sines
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
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

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<ResidualBlockW> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(ResidualBlockW {
        hidden_w:   load_t(content, reader, &p("hidden.weight"), device)?,
        hidden_b:   load_t(content, reader, &p("hidden.bias"), device)?,
        output_w:   load_t(content, reader, &p("output.weight"), device)?,
        output_b:   load_t(content, reader, &p("output.bias"), device)?,
        residual_w: load_t(content, reader, &p("residual.weight"), device)?,
        residual_b: load_t(content, reader, &p("residual.bias"), device)?,
    })
}

impl Moirai2Model {
    pub fn load(gguf_path: &Path, config: Moirai2Config) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let in_proj = load_residual_block(&content, &mut reader, "in_proj", &device)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let q_w = load_t(&content, &mut reader, &p("attn_q.weight"), &device)?;
            let k_w = load_t(&content, &mut reader, &p("attn_k.weight"), &device)?;
            let v_w = load_t(&content, &mut reader, &p("attn_v.weight"), &device)?;
            let attn_qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
                .with_context(|| format!("qkv cat blk.{n}"))?;
            let norm1_w    = load_t(&content, &mut reader, &p("norm1.weight"), &device)?;
            let norm2_w    = load_t(&content, &mut reader, &p("norm2.weight"), &device)?;
            let attn_o_w   = load_t(&content, &mut reader, &p("attn_o.weight"), &device)?;
            let attn_qn_w  = load_t(&content, &mut reader, &p("attn_qn.weight"), &device)?;
            let attn_kn_w  = load_t(&content, &mut reader, &p("attn_kn.weight"), &device)?;
            let vbias_raw = load_t(&content, &mut reader, &p("attn_vbias.weight"), &device)?
                .flatten_all()?.to_vec1::<f32>()?;
            let n_heads = config.n_heads;
            let same_var_bias: Vec<f32> = (0..n_heads).map(|h| vbias_raw[n_heads + h]).collect();
            let attn_vbias_t = Tensor::from_vec(same_var_bias, (n_heads, 1, 1), &device)?;
            let ffn_fc1_w  = load_t(&content, &mut reader, &p("ffn_fc1.weight"), &device)?;
            let ffn_fc2_w  = load_t(&content, &mut reader, &p("ffn_fc2.weight"), &device)?;
            let ffn_gate_w = load_t(&content, &mut reader, &p("ffn_gate.weight"), &device)?;
            blocks.push(EncoderBlock {
                norm1_w, norm2_w, attn_qkv_w, attn_o_w, attn_qn_w, attn_kn_w,
                attn_vbias_t, ffn_fc1_w, ffn_fc2_w, ffn_gate_w,
            });
        }

        let norm_f_w = load_t(&content, &mut reader, "norm_f.weight", &device)?;
        let out_proj = load_residual_block(&content, &mut reader, "out_proj", &device)?;

        let half_rope = config.rope_dim / 2;
        let max_pos = config.max_ctx_tokens() + 256;
        let inv_freq: Vec<f32> = (0..half_rope)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / config.rope_dim as f32))
            .collect();
        let mut rope_cos = vec![0.0f32; max_pos * half_rope];
        let mut rope_sin = vec![0.0f32; max_pos * half_rope];
        for pos in 0..max_pos {
            for i in 0..half_rope {
                let theta = pos as f32 * inv_freq[i];
                rope_cos[pos * half_rope + i] = theta.cos();
                rope_sin[pos * half_rope + i] = theta.sin();
            }
        }

        Ok(Self { device, config, in_proj, blocks, norm_f_w, out_proj,
                  rope_cos, rope_sin, causal_mask_cache: Mutex::new(HashMap::new()) })
    }

    // -----------------------------------------------------------------------
    // Forecasting
    // -----------------------------------------------------------------------

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let ps = cfg.patch_size; // 16

        // --- PackedStdScaler normalization (context only) ---
        let max_ctx_len = cfg.max_ctx_tokens() * ps;
        let ctx: &[f32] = if context.len() > max_ctx_len {
            &context[context.len() - max_ctx_len..]
        } else {
            context
        };
        let n = ctx.len() as f64;
        let loc_f64 = ctx.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = ctx.iter().map(|&v| (v as f64 - loc_f64).powi(2)).sum::<f64>()
            / (n - 1.0).max(1.0);
        let scale = ((var + 1e-5_f64).sqrt()) as f32;
        let loc   = loc_f64 as f32;

        // --- Normalize and left-pad to multiple of patch_size ---
        let ctx_norm: Vec<f32> = ctx.iter().map(|&v| (v - loc) / scale).collect();
        let rem = ctx_norm.len() % ps;
        let ctx_padded: Vec<f32> = if rem != 0 {
            let mut padded = vec![0.0f32; ps - rem];
            padded.extend_from_slice(&ctx_norm);
            padded
        } else {
            ctx_norm
        };
        let n_ctx = ctx_padded.len() / ps;

        let ctx_flat: Vec<f32> = (0..n_ctx)
            .flat_map(|i| {
                let mut tok = ctx_padded[i * ps..(i + 1) * ps].to_vec();
                tok.extend(vec![1.0f32; ps]);
                tok
            })
            .collect();
        let ctx_time_ids: Vec<usize> = (0..n_ctx).collect();

        let num_pt = cfg.num_predict_token; // 4
        let num_q  = cfg.num_quantiles;     // 9
        let mq     = cfg.median_quantile;   // 4
        let n_future_patches = (horizon + ps - 1) / ps;

        // --- Prefill: run context through transformer, collect KV cache ---
        let input_t = Tensor::from_vec(ctx_flat, (n_ctx, ps * 2), &self.device)?;
        let h_ctx = residual_block_fwd(&input_t, &self.in_proj)?;
        let (h_enc, mut kv_cache) = self.prefill_encoder(h_ctx, &ctx_time_ids, n_ctx)?;

        // Apply norm_f + out_proj to last context token only
        let last_h_norm = rms_norm(&h_enc.narrow(0, n_ctx - 1, 1)?, &self.norm_f_w)?;
        let first_pred: Vec<f32> = residual_block_fwd(&last_h_norm, &self.out_proj)?
            .flatten_all()?.to_vec1()?;

        let mut collected_patches: Vec<Vec<f32>> = Vec::new();
        let mut prev_patches: Vec<Vec<f32>> = (0..num_pt)
            .map(|pt| {
                let q_start = pt * num_q * ps + mq * ps;
                first_pred[q_start..q_start + ps].to_vec()
            })
            .collect();
        let n_take = n_future_patches.min(num_pt);
        for patch in prev_patches.iter().take(n_take) {
            collected_patches.push(patch.clone());
        }

        // --- Decode loop: num_pt=4 new tokens per step, cached K/V ---
        let mut cached_len = n_ctx;
        while collected_patches.len() < n_future_patches {
            let new_time_ids: Vec<usize> = (cached_len..cached_len + num_pt).collect();
            let new_flat: Vec<f32> = prev_patches.iter()
                .flat_map(|patch| {
                    let mut tok = patch.clone();
                    tok.extend(vec![0.0f32; ps]);
                    tok
                })
                .collect();
            let new_t = Tensor::from_vec(new_flat, (num_pt, ps * 2), &self.device)?;
            let h_in = residual_block_fwd(&new_t, &self.in_proj)?;
            let h_dec = self.decode_encoder(h_in, &new_time_ids, &mut kv_cache, cached_len)?;

            let last_h_norm = rms_norm(&h_dec.narrow(0, num_pt - 1, 1)?, &self.norm_f_w)?;
            let pred: Vec<f32> = residual_block_fwd(&last_h_norm, &self.out_proj)?
                .flatten_all()?.to_vec1()?;

            let new_patches: Vec<Vec<f32>> = (0..num_pt)
                .map(|pt| {
                    let q_start = pt * num_q * ps + mq * ps;
                    pred[q_start..q_start + ps].to_vec()
                })
                .collect();

            let need = n_future_patches - collected_patches.len();
            for patch in new_patches.iter().take(need.min(num_pt)) {
                collected_patches.push(patch.clone());
            }
            cached_len += num_pt;
            prev_patches = new_patches;
        }

        // Flatten and denormalize
        let result: Vec<f32> = collected_patches
            .iter()
            .flat_map(|p| p.iter().copied())
            .take(horizon)
            .map(|v| v * scale + loc)
            .collect();

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Prefill: full forward pass collecting per-layer K/V tensors
    // K/V stored as [n_heads, seq, head_dim]
    // -----------------------------------------------------------------------

    fn prefill_encoder(
        &self,
        mut h: Tensor,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let mut kv_cache: Vec<(Tensor, Tensor)> = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (h_out, k, v) = self.prefill_block(h, blk, time_ids, seq_len)?;
            h = h_out;
            kv_cache.push((k, v));
        }
        Ok((h, kv_cache))
    }

    fn prefill_block(
        &self,
        h: Tensor,
        blk: &EncoderBlock,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_norm = rms_norm(&h, &blk.norm1_w)?;
        let (attn_out, k, v) = self.prefill_attn(&h_norm, blk, time_ids, seq_len)?;
        let h = (h + attn_out)?;
        let h_norm2 = rms_norm(&h, &blk.norm2_w)?;
        let ffn_out = swiglu_ffn(&h_norm2, &blk.ffn_fc1_w, &blk.ffn_fc2_w, &blk.ffn_gate_w)?;
        Ok(((h + ffn_out)?, k, v))
    }

    fn prefill_attn(
        &self,
        h: &Tensor,
        blk: &EncoderBlock,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let n_heads  = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model  = cfg.d_model;
        let rope_dim = cfg.rope_dim;

        let qkv = linear_nobias(h, &blk.attn_qkv_w)?;
        let q = qkv.narrow(1, 0, d_model)?;
        let k = qkv.narrow(1, d_model, d_model)?;
        let v = qkv.narrow(1, 2 * d_model, d_model)?;

        let q = q.reshape((seq_len, n_heads, head_dim))?;
        let k = k.reshape((seq_len, n_heads, head_dim))?;
        let q = qk_norm_heads(&q, &blk.attn_qn_w, seq_len, n_heads, head_dim)?;
        let k = qk_norm_heads(&k, &blk.attn_kn_w, seq_len, n_heads, head_dim)?;

        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((seq_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        let q = apply_partial_rope(&q, time_ids, n_heads, head_dim, rope_dim, &self.device, &self.rope_cos, &self.rope_sin)?;
        let k = apply_partial_rope(&k, time_ids, n_heads, head_dim, rope_dim, &self.device, &self.rope_cos, &self.rope_sin)?;

        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k.permute((0, 2, 1))?)?;
        let scores = (scores / scale)?;
        let causal = {
            let mut cache = self.causal_mask_cache.lock().unwrap();
            if !cache.contains_key(&seq_len) {
                cache.insert(seq_len, make_causal_mask_tensor(seq_len, n_heads, &self.device)?);
            }
            cache[&seq_len].clone()
        };
        let scores = scores.broadcast_add(&causal)?;
        let scores = scores.broadcast_add(&blk.attn_vbias_t)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((seq_len, d_model))?;
        Ok((linear_nobias(&out, &blk.attn_o_w)?, k, v))
    }

    // -----------------------------------------------------------------------
    // KV-cached decode: process num_pt new tokens, append to cache
    // -----------------------------------------------------------------------

    fn decode_encoder(
        &self,
        mut h: Tensor,
        new_time_ids: &[usize],
        kv_cache: &mut Vec<(Tensor, Tensor)>,
        cached_len: usize,
    ) -> Result<Tensor> {
        let new_len = new_time_ids.len();
        for (li, blk) in self.blocks.iter().enumerate() {
            // decode_block_kv returns the extended K/V (old cache + new tokens).
            // Store directly — no second Tensor::cat needed.
            let (h_out, k_new, v_new) =
                self.decode_block_kv(h, blk, &kv_cache[li], new_time_ids, new_len, cached_len)?;
            h = h_out;
            kv_cache[li] = (k_new, v_new);
        }
        Ok(h)
    }

    fn decode_block_kv(
        &self,
        h: Tensor,
        blk: &EncoderBlock,
        cache: &(Tensor, Tensor),
        new_time_ids: &[usize],
        new_len: usize,
        cached_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let h_norm = rms_norm(&h, &blk.norm1_w)?;
        let (attn_out, k_new, v_new) =
            self.decode_attn_kv(&h_norm, blk, cache, new_time_ids, new_len, cached_len)?;
        let h = (h + attn_out)?;
        let h_norm2 = rms_norm(&h, &blk.norm2_w)?;
        let ffn_out = swiglu_ffn(&h_norm2, &blk.ffn_fc1_w, &blk.ffn_fc2_w, &blk.ffn_gate_w)?;
        Ok(((h + ffn_out)?, k_new, v_new))
    }

    fn decode_attn_kv(
        &self,
        h: &Tensor,
        blk: &EncoderBlock,
        cache: &(Tensor, Tensor),
        new_time_ids: &[usize],
        new_len: usize,
        cached_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let n_heads  = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model  = cfg.d_model;
        let rope_dim = cfg.rope_dim;

        let qkv = linear_nobias(h, &blk.attn_qkv_w)?;
        let q = qkv.narrow(1, 0, d_model)?;
        let k = qkv.narrow(1, d_model, d_model)?;
        let v = qkv.narrow(1, 2 * d_model, d_model)?;

        let q = q.reshape((new_len, n_heads, head_dim))?;
        let k = k.reshape((new_len, n_heads, head_dim))?;
        let q = qk_norm_heads(&q, &blk.attn_qn_w, new_len, n_heads, head_dim)?;
        let k = qk_norm_heads(&k, &blk.attn_kn_w, new_len, n_heads, head_dim)?;

        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((new_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        let q = apply_partial_rope(&q, new_time_ids, n_heads, head_dim, rope_dim, &self.device, &self.rope_cos, &self.rope_sin)?;
        let k = apply_partial_rope(&k, new_time_ids, n_heads, head_dim, rope_dim, &self.device, &self.rope_cos, &self.rope_sin)?;

        // Extend cache: [n_heads, cached+new_len, head_dim].
        // Return k_full/v_full so the caller can store them directly without a second cat.
        let k_full = Tensor::cat(&[&cache.0, &k], 1)?;
        let v_full = Tensor::cat(&[&cache.1, &v], 1)?;

        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k_full.permute((0, 2, 1))?)?; // [n_heads, new_len, cached+new_len]
        let scores = (scores / scale)?;
        let scores = add_decode_causal_mask(&scores, new_len, cached_len, n_heads, &self.device)?;
        let scores = scores.broadcast_add(&blk.attn_vbias_t)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v_full)?; // [n_heads, new_len, head_dim]
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((new_len, d_model))?;
        Ok((linear_nobias(&out, &blk.attn_o_w)?, k_full, v_full))
    }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

fn residual_block_fwd(x: &Tensor, w: &ResidualBlockW) -> Result<Tensor> {
    let hidden   = linear_with_bias(x, &w.hidden_w, &w.hidden_b)?.silu()?;
    let output   = linear_with_bias(&hidden, &w.output_w, &w.output_b)?;
    let residual = linear_with_bias(x, &w.residual_w, &w.residual_b)?;
    Ok((output + residual)?)
}

fn linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?)
}

fn linear_with_bias(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?.broadcast_add(b)?)
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let rms = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rms = (rms + 1e-6_f64)?.sqrt()?;
    let x = x.broadcast_div(&rms)?;
    Ok(x.broadcast_mul(weight)?)
}

fn qk_norm_heads(
    x: &Tensor,
    weight: &Tensor,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let x_flat = x.reshape((seq_len * n_heads, head_dim))?;
    let normed  = rms_norm(&x_flat, weight)?;
    Ok(normed.reshape((seq_len, n_heads, head_dim))?)
}

fn swiglu_ffn(x: &Tensor, fc1_w: &Tensor, fc2_w: &Tensor, gate_w: &Tensor) -> Result<Tensor> {
    let content = linear_nobias(x, fc1_w)?.silu()?;
    let gate    = linear_nobias(x, gate_w)?;
    let h = (content * gate)?;
    linear_nobias(&h, fc2_w)
}

/// Build lower-triangular causal mask [1, seq, seq] — broadcast_add handles head dim.
fn make_causal_mask_tensor(seq_len: usize, n_heads: usize, device: &Device) -> Result<Tensor> {
    let _ = n_heads;
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask, (1usize, seq_len, seq_len), device).map_err(anyhow::Error::from)
}

/// Decode causal mask: [n_heads, new_len, cached+new_len].
/// New token q_rel attends to all cached tokens and new tokens 0..=q_rel.
fn add_decode_causal_mask(
    scores: &Tensor,
    new_len: usize,
    cached_len: usize,
    n_heads: usize,
    device: &Device,
) -> Result<Tensor> {
    let total = cached_len + new_len;
    let mut mask = vec![0.0f32; new_len * total];
    for q_rel in 0..new_len {
        for k_abs in (cached_len + q_rel + 1)..total {
            mask[q_rel * total + k_abs] = f32::NEG_INFINITY;
        }
    }
    let mask_t = Tensor::from_vec(mask, (1usize, new_len, total), device)?;
    Ok(scores.broadcast_add(&mask_t)?)
}

