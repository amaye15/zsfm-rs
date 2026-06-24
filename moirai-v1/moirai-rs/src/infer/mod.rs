//! Moirai-1.0-R-large inference engine.
//!
//! Architecture: masked bidirectional transformer encoder with multi-scale patches.
//! - Mean-scale normalization → patch embedding (in_proj per patch size) →
//!   mask tokens for future patches → 24 encoder blocks (RMSNorm + QK-norm
//!   attention + SwiGLU FFN) → final RMSNorm → Student-t head →
//!   inverse scale → point forecast (mean = loc)

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use std::collections::HashMap;
use std::sync::Mutex;

use crate::config::MoiraiConfig;

// Chosen patch size for inference: 32 (index 2 in [8,16,32,64,128])
const PATCH_SIZE: usize = 32;
const PATCH_IDX: usize  = 2;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct EncoderBlock {
    norm1_w:         Tensor,
    norm2_w:         Tensor,
    attn_qkv_w:      Tensor, // fused [3*d_model, d_model]
    attn_o_w:        Tensor,
    attn_qn_w:       Tensor, // Q per-head norm scale [head_dim]
    attn_kn_w:       Tensor, // K per-head norm scale [head_dim]
    vbias_obs: Tensor,   // [n_heads, 1, 1] — observed-key per-head bias
    vbias_mask: Tensor,  // [n_heads, 1, 1] — masked-key per-head bias
    ffn_fc1_w:       Tensor,
    ffn_fc2_w:       Tensor,
    ffn_gate_w:      Tensor,
}

pub struct MoiraiModel {
    device: Device,
    config: MoiraiConfig,
    in_proj_w:     Tensor, // [5, 1024, 128]
    in_proj_b:     Tensor, // [5, 1024]
    mask_embed:    Tensor, // [1, 1024]
    blocks:        Vec<EncoderBlock>,
    norm_f_w:      Tensor,
    head_st_loc_w: Tensor, // [5, 128, 1024]
    head_st_loc_b: Tensor, // [5, 128]
    rope_inv_freq: Vec<f32>,
    rope_cache:    Mutex<HashMap<usize, (Tensor, Tensor)>>,
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

impl MoiraiModel {
    pub fn load(gguf_path: &Path, config: MoiraiConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let in_proj_w  = load_t(&content, &mut reader, "in_proj.weight", &device)?;
        let in_proj_b  = load_t(&content, &mut reader, "in_proj.bias", &device)?;
        let mask_embed = load_t(&content, &mut reader, "mask_embed.weight", &device)?;

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
            let vbias_obs  = Tensor::from_vec(vbias_raw[0..n_heads].to_vec(), (n_heads, 1, 1), &device)?;
            let vbias_mask = Tensor::from_vec(vbias_raw[n_heads..2*n_heads].to_vec(), (n_heads, 1, 1), &device)?;
            let ffn_fc1_w  = load_t(&content, &mut reader, &p("ffn_fc1.weight"), &device)?;
            let ffn_fc2_w  = load_t(&content, &mut reader, &p("ffn_fc2.weight"), &device)?;
            let ffn_gate_w = load_t(&content, &mut reader, &p("ffn_gate.weight"), &device)?;
            blocks.push(EncoderBlock {
                norm1_w, norm2_w, attn_qkv_w, attn_o_w, attn_qn_w, attn_kn_w,
                vbias_obs, vbias_mask, ffn_fc1_w, ffn_fc2_w, ffn_gate_w,
            });
        }

        let norm_f_w      = load_t(&content, &mut reader, "norm_f.weight", &device)?;
        let head_st_loc_w = load_t(&content, &mut reader, "head.st_loc.weight", &device)?;
        let head_st_loc_b = load_t(&content, &mut reader, "head.st_loc.bias", &device)?;

        let head_dim = config.head_dim;
        let half = head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        Ok(Self {
            device, config,
            in_proj_w, in_proj_b, mask_embed,
            blocks, norm_f_w,
            head_st_loc_w, head_st_loc_b,
            rope_inv_freq,
            rope_cache: Mutex::new(HashMap::new()),
        })
    }

    // -----------------------------------------------------------------------
    // Forecasting
    // -----------------------------------------------------------------------

    /// Zero-shot forecasting: embed context + masked future, single encoder pass.
    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let patch_size = PATCH_SIZE;
        let patch_idx  = PATCH_IDX;

        // Mean-scale normalization (Moirai's default scaling)
        let loc   = context.iter().map(|&v| v as f64).sum::<f64>() / context.len() as f64;
        let scale = context.iter().map(|&v| (v as f64 - loc).abs()).sum::<f64>()
                    / context.len() as f64;
        let scale = (scale.max(1e-8)) as f32;
        let loc   = loc as f32;

        // Scale context and trim/pad to max_seq_len
        let max_ts = cfg.max_seq_len; // 512
        let mut ctx_scaled: Vec<f32> = context.iter().map(|&v| (v - loc) / scale).collect();
        if ctx_scaled.len() > max_ts {
            let start = ctx_scaled.len() - max_ts;
            ctx_scaled = ctx_scaled[start..].to_vec();
        }
        // Pad to next multiple of patch_size
        let ctx_len = ctx_scaled.len();
        let ctx_padded_len = ((ctx_len + patch_size - 1) / patch_size) * patch_size;
        if ctx_padded_len > ctx_len {
            let mut padded = vec![0.0f32; ctx_padded_len - ctx_len];
            padded.extend_from_slice(&ctx_scaled);
            ctx_scaled = padded;
        }
        let n_ctx_patches = ctx_scaled.len() / patch_size;

        // Number of future patches
        let n_fc_patches = (horizon + patch_size - 1) / patch_size;
        let total_patches = n_ctx_patches + n_fc_patches;

        // Embed context patches: in_proj_w[patch_idx] is [1024, 128], use [:, :patch_size]
        // in_proj shape: [5, 1024, 128] stored as GGUF [128, 1024, 5] (reversed)
        // After dequantize candle gives us the tensor in its stored order
        // The Python shape [5, 1024, 128] → GGUF reversal → [128, 1024, 5]
        // We need to work with this carefully.
        let d_model      = cfg.d_model;       // 1024
        let max_ps       = cfg.max_patch_size; // 128

        // Python in_proj.weight[patch_idx]: [d_model=1024, max_ps=128]
        // In GGUF (reversed dims): the tensor is stored as [max_ps=128, d_model=1024, 5_dim]?
        // Actually, for a 3D tensor [5, 1024, 128], GGUF stores shape in Python order but
        // Candle dequantizes preserving the original shape. Let's reshape to [5, 1024, 128].
        let in_proj_w_3d = self.in_proj_w.reshape((5, d_model, max_ps))?;
        // Extract patch_idx slice: [d_model, max_ps]
        let proj_w_slice = in_proj_w_3d.get(patch_idx)?.contiguous()?; // [d_model, max_ps]
        // Take only patch_size columns: [d_model, patch_size]
        let proj_w = proj_w_slice.narrow(1, 0, patch_size)?.contiguous()?;

        // Bias: in_proj_b[patch_idx] → [d_model=1024]
        let in_proj_b_2d = self.in_proj_b.reshape((5, d_model))?;
        let proj_b = in_proj_b_2d.get(patch_idx)?.contiguous()?; // [d_model]

        // Build patch embeddings for context: [n_ctx_patches, d_model]
        let mut patch_flat = vec![0.0f32; n_ctx_patches * patch_size];
        for i in 0..n_ctx_patches {
            let src = &ctx_scaled[i * patch_size..(i + 1) * patch_size];
            patch_flat[i * patch_size..(i + 1) * patch_size].copy_from_slice(src);
        }
        let ctx_patches_t = Tensor::from_vec(
            patch_flat, (n_ctx_patches, patch_size), &self.device,
        )?;
        // [n_ctx, patch_size] @ [patch_size, d_model] + [d_model] → [n_ctx, d_model]
        let ctx_emb = ctx_patches_t
            .matmul(&proj_w.t()?)?
            .broadcast_add(&proj_b)?;

        // Mask embeddings for future patches: tile mask_embed [n_fc, d_model]
        // mask_embed shape in GGUF: [1024, 1] (Python [1, 1024] → reversed)
        let mask_embed = self.mask_embed.reshape((1, d_model))?;
        let fc_emb = mask_embed.expand((n_fc_patches, d_model))?;

        // Concatenate: [total_patches, d_model]
        let mut h = Tensor::cat(&[&ctx_emb, &fc_emb], 0)?;

        // Build is_masked indicator for var_attn_bias: [total_patches]
        // 0 = context, 1 = masked future
        let mut is_masked = vec![0u8; total_patches];
        for i in n_ctx_patches..total_patches {
            is_masked[i] = 1;
        }

        // Encoder
        for blk in &self.blocks {
            h = self.forward_block(&h, blk, &is_masked, total_patches)?;
        }
        h = rms_norm(&h, &self.norm_f_w)?;

        // Apply Student-t loc head to future positions
        // head_st_loc_w shape: Python [5, 128, 1024] → GGUF [1024, 128, 5]
        // After reshape: [5, 128, 1024]
        let loc_w_3d = self.head_st_loc_w.reshape((5, max_ps, d_model))?;
        let loc_b_2d = self.head_st_loc_b.reshape((5, max_ps))?;
        let loc_w = loc_w_3d.get(patch_idx)?.narrow(0, 0, patch_size)?.contiguous()?; // [patch_size, d_model]
        let loc_b = loc_b_2d.get(patch_idx)?.narrow(0, 0, patch_size)?.contiguous()?; // [patch_size]

        let future_h = h.narrow(0, n_ctx_patches, n_fc_patches)?; // [n_fc, d_model]
        // [n_fc, d_model] @ [d_model, patch_size] + [patch_size] → [n_fc, patch_size]
        let pred = future_h.matmul(&loc_w.t()?)?.broadcast_add(&loc_b)?;

        let pred_flat: Vec<f32> = pred.flatten_all()?.to_vec1()?;
        let result: Vec<f32> = pred_flat
            .iter()
            .take(horizon)
            .map(|&v| v * scale + loc)
            .collect();

        Ok(result)
    }

    fn forward_block(
        &self,
        hidden: &Tensor,
        blk: &EncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor> {
        let res = hidden;
        let h = rms_norm(hidden, &blk.norm1_w)?;
        let h = self.qk_attn(&h, blk, is_masked, seq_len)?;
        let h = (h + res)?;

        let res2 = h.clone();
        let h2 = rms_norm(&h, &blk.norm2_w)?;
        let h2 = swiglu_ffn(&h2, &blk.ffn_fc1_w, &blk.ffn_fc2_w, &blk.ffn_gate_w)?;
        Ok((h2 + res2)?)
    }

    fn qk_attn(
        &self,
        hidden: &Tensor,
        blk: &EncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let n_heads  = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model  = cfg.d_model;

        let qkv = linear_nobias(hidden, &blk.attn_qkv_w)?;
        let q = qkv.narrow(D::Minus1, 0, d_model)?;
        let k = qkv.narrow(D::Minus1, d_model, d_model)?;
        let v = qkv.narrow(D::Minus1, 2 * d_model, d_model)?;

        // Reshape: [seq, n_heads, head_dim]
        let q = q.reshape((seq_len, n_heads, head_dim))?;
        let k = k.reshape((seq_len, n_heads, head_dim))?;

        // QK per-head norm then RoPE
        let q = qk_norm_heads(&q, &blk.attn_qn_w, seq_len, n_heads, head_dim)?;
        let k = qk_norm_heads(&k, &blk.attn_kn_w, seq_len, n_heads, head_dim)?;

        // [seq, n_heads, head_dim] → [n_heads, seq, head_dim]
        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;

        // Apply RoPE for positional encoding (cached by seq_len)
        let (cos_t, sin_t) = {
            let mut cache = self.rope_cache.lock().unwrap();
            if !cache.contains_key(&seq_len) {
                let half = self.rope_inv_freq.len();
                let mut cos_v = vec![0.0f32; seq_len * half];
                let mut sin_v = vec![0.0f32; seq_len * half];
                for pos in 0..seq_len {
                    for i in 0..half {
                        let theta = pos as f32 * self.rope_inv_freq[i];
                        cos_v[pos * half + i] = theta.cos();
                        sin_v[pos * half + i] = theta.sin();
                    }
                }
                let half_dim = self.rope_inv_freq.len();
                let cos_t = Tensor::from_vec(cos_v, (seq_len, half_dim), &self.device)?.unsqueeze(0)?;
                let sin_t = Tensor::from_vec(sin_v, (seq_len, half_dim), &self.device)?.unsqueeze(0)?;
                cache.insert(seq_len, (cos_t, sin_t));
            }
            let (c, s) = &cache[&seq_len];
            (c.clone(), s.clone())
        };

        let q = apply_rope_with_tables(&q, &cos_t, &sin_t, head_dim)?;
        let k = apply_rope_with_tables(&k, &cos_t, &sin_t, head_dim)?;
        let v = v.reshape((seq_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k.permute((0, 2, 1))?)?;  // [n_heads, seq, seq]
        let scores = (scores / scale)?;

        // Add variate attention bias using precomputed vbias data
        let scores = apply_var_attn_bias(&scores, is_masked, seq_len, &blk.vbias_obs, &blk.vbias_mask, &self.device)?;

        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?; // [n_heads, seq, head_dim]
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((seq_len, d_model))?;

        linear_nobias(&out, &blk.attn_o_w)
    }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

fn linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?)
}

/// RMSNorm (eps=1e-6 matching Moirai).
fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let rms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (rms + 1e-6_f64)?.sqrt()?;
    let x = x.broadcast_div(&rms)?;
    Ok(x.broadcast_mul(weight)?)
}

/// QK per-head RMSNorm: normalize each head independently.
/// x: [seq, n_heads, head_dim] → normalize over head_dim per head
fn qk_norm_heads(
    x: &Tensor,
    weight: &Tensor, // [head_dim]
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let x_flat = x.reshape((seq_len * n_heads, head_dim))?;
    let normed = rms_norm(&x_flat, weight)?;
    Ok(normed.reshape((seq_len, n_heads, head_dim))?)
}

/// SwiGLU FFN: out = fc2(silu(fc1(x)) * fc_gate(x))
fn swiglu_ffn(x: &Tensor, fc1_w: &Tensor, fc2_w: &Tensor, gate_w: &Tensor) -> Result<Tensor> {
    let content = linear_nobias(x, fc1_w)?.silu()?;
    let gate    = linear_nobias(x, gate_w)?;
    let h = (content * gate)?;
    linear_nobias(&h, fc2_w)
}

/// Add variate attention bias to scores [n_heads, seq, seq].
/// Build [n_heads, 1, seq_len] variate bias via per-key mask selection:
///   bias[head, key] = vbias_mask[head] if is_masked[key] else vbias_obs[head]
///                   = vbias_obs[head] + (vbias_mask[head] - vbias_obs[head]) * m[key]
fn apply_var_attn_bias(
    scores: &Tensor,
    is_masked: &[u8],
    seq_len: usize,
    vbias_obs: &Tensor,   // [n_heads, 1, 1]
    vbias_mask: &Tensor,  // [n_heads, 1, 1]
    device: &Device,
) -> Result<Tensor> {
    let mask_f: Vec<f32> = is_masked.iter().map(|&m| m as f32).collect();
    let mask_t = Tensor::from_vec(mask_f, (1usize, 1, seq_len), device)?;
    let delta = (vbias_mask - vbias_obs)?;
    let bias = vbias_obs.broadcast_add(&delta.broadcast_mul(&mask_t)?)?;
    Ok(scores.broadcast_add(&bias)?)
}

/// RoPE using precomputed cos/sin tables [1, seq, half].
/// x is [n_heads, seq, head_dim].
fn apply_rope_with_tables(
    x: &Tensor,
    cos_t: &Tensor,
    sin_t: &Tensor,
    head_dim: usize,
) -> Result<Tensor> {
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?.contiguous()?;
    let x2 = x.narrow(D::Minus1, half, half)?.contiguous()?;
    let rot1 = (x1.broadcast_mul(cos_t)? - x2.broadcast_mul(sin_t)?)?;
    let rot2 = (x1.broadcast_mul(sin_t)? + x2.broadcast_mul(cos_t)?)?;
    Ok(Tensor::cat(&[&rot1, &rot2], D::Minus1)?.contiguous()?)
}
