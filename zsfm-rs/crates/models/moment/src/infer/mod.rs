//! MOMENT-1-large inference engine.
//!
//! Architecture: T5 encoder-only transformer with patch embeddings.
//! - RevIN normalization → patch embedding (value + position) →
//!   24 T5 encoder blocks (RMSNorm + relative-bias MHA + gated-GELU FFN) →
//!   final RMSNorm → per-patch reconstruction head
//! - Zero-shot forecasting: iteratively predict 8 steps at a time using
//!   the reconstruction head applied to the last patch's encoder output.

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use crate::config::MomentConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct EncoderBlock {
    attn_qkv_w: Tensor, // fused [3*d_model, d_model]
    attn_o_w: Tensor,
    attn_norm_w: Tensor,
    ffn_wi0_w: Tensor,
    ffn_wi1_w: Tensor,
    ffn_wo_w: Tensor,
    ffn_norm_w: Tensor,
}

pub struct MomentModel {
    device: Device,
    config: MomentConfig,
    patch_embed_w: Tensor,    // [1024, 8]  value embedding (no bias)
    pos_embed: Tensor,        // [1, 5000, 1024]
    mask_embed: Tensor,       // [1024]  learned token for masked (future) patches
    rel_bias_data: Vec<f32>,  // [32 * 16] flat, precomputed from rel_bias_w
    blocks: Vec<EncoderBlock>,
    norm_f_w: Tensor,
    head_w: Tensor,           // [8, 1024]
    head_b: Tensor,           // [8]
    rel_bias_cache: Mutex<HashMap<usize, Tensor>>,
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
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

impl MomentModel {
    pub fn load(gguf_path: &Path, config: MomentConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let patch_embed_w = load_t(&content, &mut reader, "patch_embed.weight", &device)?;
        let pos_embed     = load_t(&content, &mut reader, "pos_embed.pe", &device)?;
        let mask_embed    = load_t(&content, &mut reader, "mask_embed", &device)?;
        let rel_bias_data = load_t(&content, &mut reader, "blk.0.attn_rel_bias.weight", &device)?
            .flatten_all()?.to_vec1::<f32>()?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let q_w = load_t(&content, &mut reader, &p("attn_q.weight"), &device)?;
            let k_w = load_t(&content, &mut reader, &p("attn_k.weight"), &device)?;
            let v_w = load_t(&content, &mut reader, &p("attn_v.weight"), &device)?;
            let attn_qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
                .with_context(|| format!("qkv cat blk.{n}"))?;
            blocks.push(EncoderBlock {
                attn_qkv_w,
                attn_o_w:   load_t(&content, &mut reader, &p("attn_o.weight"), &device)?,
                attn_norm_w: load_t(&content, &mut reader, &p("attn_norm.weight"), &device)?,
                ffn_wi0_w:  load_t(&content, &mut reader, &p("ffn_wi0.weight"), &device)?,
                ffn_wi1_w:  load_t(&content, &mut reader, &p("ffn_wi1.weight"), &device)?,
                ffn_wo_w:   load_t(&content, &mut reader, &p("ffn_wo.weight"), &device)?,
                ffn_norm_w: load_t(&content, &mut reader, &p("ffn_norm.weight"), &device)?,
            });
        }

        let norm_f_w = load_t(&content, &mut reader, "norm_f.weight", &device)?;
        let head_w   = load_t(&content, &mut reader, "head.weight", &device)?;
        let head_b   = load_t(&content, &mut reader, "head.bias", &device)?;

        Ok(Self {
            device, config,
            patch_embed_w, pos_embed, mask_embed, rel_bias_data,
            blocks, norm_f_w, head_w, head_b,
            rel_bias_cache: Mutex::new(HashMap::new()),
        })
    }

    // -----------------------------------------------------------------------
    // Forecasting
    // -----------------------------------------------------------------------

    /// Zero-shot forecasting via single-pass masked inpainting.
    ///
    /// Appends `n_forecast_patches` zero-filled future patches after the 64
    /// context patches and runs ONE T5 encoder pass. The bidirectional encoder
    /// fills the masked future positions using the real context, mirroring
    /// the masking pretraining task. Head applied to future positions gives
    /// the forecast.
    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let patch_len = cfg.patch_len;
        let seq_len = cfg.seq_len;
        let ctx_patches = seq_len / cfg.patch_stride; // 64

        // RevIN normalization
        let (loc, scale) = revin_stats(context);
        let scale = scale.max(1e-8);

        // Scale and pad/truncate context to seq_len
        let mut ctx_scaled: Vec<f32> = context.iter().map(|&v| (v - loc) / scale).collect();
        if ctx_scaled.len() < seq_len {
            let pad = seq_len - ctx_scaled.len();
            let mut padded = vec![0.0f32; pad];
            padded.extend_from_slice(&ctx_scaled);
            ctx_scaled = padded;
        } else if ctx_scaled.len() > seq_len {
            let start = ctx_scaled.len() - seq_len;
            ctx_scaled = ctx_scaled[start..].to_vec();
        }

        // Number of future patches to generate
        let n_fc = (horizon + patch_len - 1) / patch_len;
        let total_patches = ctx_patches + n_fc;

        let rel_bias = {
            let mut cache = self.rel_bias_cache.lock().unwrap();
            if !cache.contains_key(&total_patches) {
                cache.insert(total_patches, self.compute_rel_bias(total_patches)?);
            }
            cache[&total_patches].clone()
        };

        // Single encoder pass: context patches + mask_embed for future positions
        let ctx_patches_data = patchify(&ctx_scaled, patch_len, cfg.patch_stride, ctx_patches);
        let mut h = self.embed_patches_with_mask(&ctx_patches_data, n_fc, total_patches)?;
        for blk in &self.blocks {
            h = self.forward_block(&h, blk, &rel_bias, total_patches)?;
        }
        h = zsfm_nn::rms_norm(&h, Some(&self.norm_f_w), cfg.layer_norm_eps)?;

        // Apply head to future positions
        let future_h = h.narrow(0, ctx_patches, n_fc)?; // [n_fc, 1024]
        let pred = zsfm_nn::linear_bias(&future_h, &self.head_w, &self.head_b)?; // [n_fc, 8]

        let pred_flat: Vec<f32> = pred.flatten_all()?.to_vec1()?;
        let result: Vec<f32> = pred_flat
            .iter()
            .take(horizon)
            .map(|&v| v * scale + loc)
            .collect();

        Ok(result)
    }

    /// Embed context patches via value_embedding, then append n_fc mask tokens.
    fn embed_patches_with_mask(
        &self,
        ctx_patches: &[Vec<f32>],
        n_fc: usize,
        total_patches: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let patch_len = cfg.patch_len;
        let d_model = cfg.d_model;
        let ctx_n = ctx_patches.len();

        // Context: value_embedding(patch) → [ctx_n, d_model]
        let flat: Vec<f32> = ctx_patches.iter().flatten().copied().collect();
        let x = Tensor::from_vec(flat, (ctx_n, patch_len), &self.device)?;
        let h_ctx = x.matmul(&self.patch_embed_w.t()?)?;

        // Future: expand mask_embed [d_model] → [n_fc, d_model]
        let h_fc = self.mask_embed.unsqueeze(0)?.broadcast_as((n_fc, d_model))?;

        // Concatenate: [total_patches, d_model]
        let h = Tensor::cat(&[h_ctx, h_fc], 0)?;

        // Position embedding: pe is [1, 5000, d_model]; take [:total_patches]
        let pos = self.pos_embed.squeeze(0)?.narrow(0, 0, total_patches)?;

        Ok((h + pos)?)
    }

    fn forward_block(
        &self,
        hidden: &Tensor,
        blk: &EncoderBlock,
        rel_bias: &Tensor, // [1, n_heads, seq, seq]
        seq_len: usize,
    ) -> Result<Tensor> {
        // Pre-norm attention (with residual)
        let res = hidden;
        let h = zsfm_nn::rms_norm(hidden, Some(&blk.attn_norm_w), self.config.layer_norm_eps)?;
        let h = self.t5_self_attn(&h, blk, rel_bias, seq_len)?;
        let h = (h + res)?;

        // Pre-norm FFN (with residual)
        let res2 = h.clone();
        let h2 = zsfm_nn::rms_norm(&h, Some(&blk.ffn_norm_w), self.config.layer_norm_eps)?;
        let h2 = gated_gelu_ffn(&h2, &blk.ffn_wi0_w, &blk.ffn_wi1_w, &blk.ffn_wo_w)?;
        Ok((h2 + res2)?)
    }

    fn t5_self_attn(
        &self,
        hidden: &Tensor,
        blk: &EncoderBlock,
        rel_bias: &Tensor, // [1, n_heads, seq, seq]
        seq_len: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model = cfg.d_model;

        // Project: single fused matmul → [seq, 3*d_model]
        let qkv = zsfm_nn::linear_nobias(hidden, &blk.attn_qkv_w)?;
        let q = qkv.narrow(D::Minus1, 0, d_model)?;
        let k = qkv.narrow(D::Minus1, d_model, d_model)?;
        let v = qkv.narrow(D::Minus1, 2 * d_model, d_model)?;

        // Reshape to [n_heads, seq, head_dim]
        let q = q.reshape((seq_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let k = k.reshape((seq_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((seq_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        // T5 does NOT scale by sqrt(head_dim) — scaling is absorbed into weight init
        let scores = q.matmul(&k.permute((0, 2, 1))?)?; // [n_heads, seq, seq]
        // Add relative position bias (broadcast over batch dim)
        let rel_bias_squeezed = rel_bias.squeeze(0)?; // [n_heads, seq, seq]
        let scores = (scores + rel_bias_squeezed)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted sum
        let out = attn.matmul(&v)?; // [n_heads, seq, head_dim]
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((seq_len, d_model))?;

        zsfm_nn::linear_nobias(&out, &blk.attn_o_w)
    }

    /// Compute T5 relative position bias for a given sequence length.
    fn compute_rel_bias(&self, seq_len: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let n_heads = cfg.n_heads;
        let num_buckets = cfg.rel_attn_num_buckets; // 32
        let max_distance = cfg.rel_attn_max_distance; // 128

        let mut out = vec![0.0f32; n_heads * seq_len * seq_len];
        for query_pos in 0..seq_len {
            for key_pos in 0..seq_len {
                let rel = key_pos as i64 - query_pos as i64;
                let bucket = t5_relative_bucket(rel, true, num_buckets, max_distance);
                for head in 0..n_heads {
                    // bias_data: row=bucket, col=head → index = bucket*n_heads + head
                    let bias_val = self.rel_bias_data[bucket * n_heads + head];
                    out[head * seq_len * seq_len + query_pos * seq_len + key_pos] = bias_val;
                }
            }
        }

        // Shape: [1, n_heads, seq, seq]
        Ok(Tensor::from_vec(out, (1, n_heads, seq_len, seq_len), &self.device)?)
    }
}

// ---------------------------------------------------------------------------
// T5 relative position bucket
// ---------------------------------------------------------------------------

fn t5_relative_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut ret = 0usize;
    let mut num_buckets = num_buckets;

    let n: usize = if bidirectional {
        num_buckets /= 2;  // each direction gets half the buckets
        if relative_position > 0 {
            ret += num_buckets;  // positive offset for future positions
        }
        relative_position.unsigned_abs() as usize
    } else {
        (-relative_position).max(0) as usize
    };

    let max_exact = num_buckets / 2;

    if n < max_exact {
        ret += n;
    } else {
        let val = max_exact
            + ((n as f32 / max_exact as f32).ln()
                / (max_distance as f32 / max_exact as f32).ln()
                * (num_buckets - max_exact) as f32) as usize;
        ret += val.min(num_buckets - 1);
    }

    ret
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

/// T5 gated-GELU FFN: out = wo(gelu(wi_0(x)) * wi_1(x))
fn gated_gelu_ffn(
    x: &Tensor,
    wi0_w: &Tensor,
    wi1_w: &Tensor,
    wo_w: &Tensor,
) -> Result<Tensor> {
    let gate  = zsfm_nn::linear_nobias(x, wi0_w)?.gelu_erf()?;
    let value = zsfm_nn::linear_nobias(x, wi1_w)?;
    let h = (gate * value)?;
    zsfm_nn::linear_nobias(&h, wo_w)
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Compute mean and std of x for RevIN normalization.
fn revin_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f64;
    if n == 0.0 { return (0.0, 1.0); }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var  = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean as f32, var.sqrt() as f32)
}

/// Extract non-overlapping patches from a sequence.
fn patchify(x: &[f32], patch_len: usize, stride: usize, num_patches: usize) -> Vec<Vec<f32>> {
    (0..num_patches)
        .map(|i| {
            let start = i * stride;
            let end = (start + patch_len).min(x.len());
            let mut patch = x[start..end].to_vec();
            patch.resize(patch_len, 0.0);
            patch
        })
        .collect()
}

// ---------------------------------------------------------------------------
// zsfm-core::Forecaster
// ---------------------------------------------------------------------------

impl zsfm_core::Forecaster for MomentModel {
    type Config = MomentConfig;

    fn load(gguf_path: &Path, config: MomentConfig) -> Result<Self> {
        MomentModel::load(gguf_path, config)
    }

    /// MOMENT is univariate-only and point-forecast-only; `mask` is unused.
    fn forecast(
        &self,
        context: &[Vec<f32>],
        _mask: &[Vec<bool>],
        horizon: usize,
    ) -> Result<zsfm_core::QuantileMatrix> {
        anyhow::ensure!(context.len() == 1, "MomentModel only supports univariate forecasting (1 variate)");
        let point = MomentModel::forecast(self, &context[0], horizon)?;
        Ok(vec![vec![point]])
    }
}
