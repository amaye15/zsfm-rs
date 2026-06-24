//! TinyTimeMixer inference engine.
//!
//! Architecture: MLP-Mixer with adaptive patching.
//! - StdScaler → Patchify → Linear patcher → 3 adaptive levels → Linear adapter →
//!   2 decoder layers → Flatten → Linear head → inverse scale
//! - No attention. Each "mixer layer" = PatchMixerBlock + FeatureMixerBlock.
//! - GatedAttention: softmax(linear(x)) * x applied after each MLP.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use crate::config::TtmConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct MixerLayer {
    patch_norm_w: Tensor,
    patch_norm_b: Tensor,
    patch_fc1_w: Tensor,
    patch_fc1_b: Tensor,
    patch_fc2_w: Tensor,
    patch_fc2_b: Tensor,
    patch_gate_w: Tensor,
    patch_gate_b: Tensor,
    feat_norm_w: Tensor,
    feat_norm_b: Tensor,
    feat_fc1_w: Tensor,
    feat_fc1_b: Tensor,
    feat_fc2_w: Tensor,
    feat_fc2_b: Tensor,
    feat_gate_w: Tensor,
    feat_gate_b: Tensor,
}

struct AdaptiveLevel {
    factor: usize,
    layers: Vec<MixerLayer>,
}

pub struct TtmModel {
    device: Device,
    config: TtmConfig,
    patcher_w: Tensor,
    patcher_b: Tensor,
    enc_levels: Vec<AdaptiveLevel>,
    dec_adapter_w: Tensor,
    dec_adapter_b: Tensor,
    dec_layers: Vec<MixerLayer>,
    head_w: Tensor,
    head_b: Tensor,
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

fn load_mixer_layer(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<MixerLayer> {
    let mut t = |s: &str| -> Result<Tensor> {
        load_t(content, reader, &format!("{prefix}.{s}"), device)
    };
    Ok(MixerLayer {
        patch_norm_w: t("patch_norm.weight")?,
        patch_norm_b: t("patch_norm.bias")?,
        patch_fc1_w:  t("patch_fc1.weight")?,
        patch_fc1_b:  t("patch_fc1.bias")?,
        patch_fc2_w:  t("patch_fc2.weight")?,
        patch_fc2_b:  t("patch_fc2.bias")?,
        patch_gate_w: t("patch_gate.weight")?,
        patch_gate_b: t("patch_gate.bias")?,
        feat_norm_w:  t("feat_norm.weight")?,
        feat_norm_b:  t("feat_norm.bias")?,
        feat_fc1_w:   t("feat_fc1.weight")?,
        feat_fc1_b:   t("feat_fc1.bias")?,
        feat_fc2_w:   t("feat_fc2.weight")?,
        feat_fc2_b:   t("feat_fc2.bias")?,
        feat_gate_w:  t("feat_gate.weight")?,
        feat_gate_b:  t("feat_gate.bias")?,
    })
}

impl TtmModel {
    pub fn load(gguf_path: &Path, config: TtmConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let patcher_w = load_t(&content, &mut reader, "enc.patcher.weight", &device)?;
        let patcher_b = load_t(&content, &mut reader, "enc.patcher.bias", &device)?;

        let n_levels = config.adaptive_patching_levels;
        let n_enc_layers = config.num_layers;
        let mut enc_levels = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            // mixers[0] was created with adapt_patch_level = n_levels-1, factor = 2^(n_levels-1)
            let factor = 1usize << (n_levels - 1 - l);
            let mut layers = Vec::with_capacity(n_enc_layers);
            for n in 0..n_enc_layers {
                let prefix = format!("enc.blk.{l}.layer.{n}");
                layers.push(load_mixer_layer(&content, &mut reader, &prefix, &device)?);
            }
            enc_levels.push(AdaptiveLevel { factor, layers });
        }

        let dec_adapter_w = load_t(&content, &mut reader, "dec.adapter.weight", &device)?;
        let dec_adapter_b = load_t(&content, &mut reader, "dec.adapter.bias", &device)?;

        let n_dec_layers = config.decoder_num_layers;
        let mut dec_layers = Vec::with_capacity(n_dec_layers);
        for n in 0..n_dec_layers {
            let prefix = format!("dec.blk.{n}");
            dec_layers.push(load_mixer_layer(&content, &mut reader, &prefix, &device)?);
        }

        let head_w = load_t(&content, &mut reader, "head.weight", &device)?;
        let head_b = load_t(&content, &mut reader, "head.bias", &device)?;

        Ok(Self {
            device,
            config,
            patcher_w,
            patcher_b,
            enc_levels,
            dec_adapter_w,
            dec_adapter_b,
            dec_layers,
            head_w,
            head_b,
        })
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Forecast univariate time series.
    ///
    /// `context` must be at least `config.patch_length` long.
    /// Returns `config.prediction_length` forecast values.
    pub fn forecast(&self, context: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;

        // 1. StdScaler
        let (scaled, mean, std) = std_scale(context);

        // 2. Patchify → [num_patches, patch_length]
        let patches = patchify(&scaled, cfg.patch_length, cfg.patch_stride, cfg.num_patches);
        let flat: Vec<f32> = patches.into_iter().flatten().collect();
        let mut h = Tensor::from_vec(flat, (cfg.num_patches, cfg.patch_length), &self.device)?;

        // 3. Patcher Linear(patch_length → d_model) → [num_patches, d_model]
        h = linear(&h, &self.patcher_w, Some(&self.patcher_b))?;

        // 4. Encoder adaptive patching (3 levels)
        for level in &self.enc_levels {
            h = self.forward_adaptive_level(&h, level)?;
        }

        // 5. Decoder adapter Linear(d_model → decoder_d_model) → [num_patches, decoder_d_model]
        h = linear(&h, &self.dec_adapter_w, Some(&self.dec_adapter_b))?;

        // 6. Decoder block (regular mixer layers)
        for layer in &self.dec_layers {
            h = forward_mixer_layer(&h, layer, cfg.norm_eps)?;
        }

        // 7. Flatten → [num_patches * decoder_d_model]
        let flat_dim = cfg.num_patches * cfg.decoder_d_model;
        h = h.reshape((flat_dim,))?;

        // 8. Head Linear(flat_dim → prediction_length) → [prediction_length]
        h = h.unsqueeze(0)?; // [1, flat_dim]
        h = linear(&h, &self.head_w, Some(&self.head_b))?; // [1, prediction_length]
        h = h.squeeze(0)?; // [prediction_length]

        // 9. Inverse scale
        let forecast: Vec<f32> = h.to_vec1()?;
        Ok(forecast.iter().map(|&v| v * std + mean).collect())
    }

    fn forward_adaptive_level(&self, hidden: &Tensor, level: &AdaptiveLevel) -> Result<Tensor> {
        let factor = level.factor;
        let (p, f) = (hidden.dim(0)?, hidden.dim(1)?);

        // Reshape: [P, F] → [P*factor, F/factor]
        let mut h = if factor > 1 {
            hidden.reshape((p * factor, f / factor))?
        } else {
            hidden.clone()
        };

        for layer in &level.layers {
            h = forward_mixer_layer(&h, layer, self.config.norm_eps)?;
        }

        // Reshape back: [P*factor, F/factor] → [P, F]
        if factor > 1 { Ok(h.reshape((p, f))?) } else { Ok(h) }
    }
}

// ---------------------------------------------------------------------------
// MLP-Mixer forward
// ---------------------------------------------------------------------------

fn forward_mixer_layer(hidden: &Tensor, layer: &MixerLayer, norm_eps: f64) -> Result<Tensor> {
    let h = forward_patch_mixer(hidden, layer, norm_eps)?;
    forward_feat_mixer(&h, layer, norm_eps)
}

/// PatchMixerBlock: normalize features, transpose, MLP on patch dim, gate, transpose back.
fn forward_patch_mixer(hidden: &Tensor, w: &MixerLayer, eps: f64) -> Result<Tensor> {
    // LayerNorm on feature dim (last dim)
    let h = layer_norm(hidden, &w.patch_norm_w, &w.patch_norm_b, eps)?;
    // Transpose [P, F] → [F, P]
    let h = h.t()?.contiguous()?;
    // MLP on last dim (P)
    let h = linear(&h, &w.patch_fc1_w, Some(&w.patch_fc1_b))?.gelu_erf()?;
    let h = linear(&h, &w.patch_fc2_w, Some(&w.patch_fc2_b))?;
    // Gated attention: softmax(linear(h)) * h
    let gate = candle_nn::ops::softmax_last_dim(
        &linear(&h, &w.patch_gate_w, Some(&w.patch_gate_b))?
    )?;
    let h = (h * gate)?;
    // Transpose back [F, P] → [P, F]
    let h = h.t()?.contiguous()?;
    // Residual
    Ok((h + hidden)?)
}

/// FeatureMixerBlock: normalize features, MLP on feature dim, gate, add residual.
fn forward_feat_mixer(hidden: &Tensor, w: &MixerLayer, eps: f64) -> Result<Tensor> {
    let h = layer_norm(hidden, &w.feat_norm_w, &w.feat_norm_b, eps)?;
    let h = linear(&h, &w.feat_fc1_w, Some(&w.feat_fc1_b))?.gelu_erf()?;
    let h = linear(&h, &w.feat_fc2_w, Some(&w.feat_fc2_b))?;
    let gate = candle_nn::ops::softmax_last_dim(
        &linear(&h, &w.feat_gate_w, Some(&w.feat_gate_b))?
    )?;
    let h = (h * gate)?;
    Ok((h + hidden)?)
}

// ---------------------------------------------------------------------------
// Primitive ops
// ---------------------------------------------------------------------------

/// y = x @ w^T + b.  Supports 2D and 3D x.
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let out = x.matmul(&w.t()?)?;
    if let Some(b) = b { Ok(out.broadcast_add(b)?) } else { Ok(out) }
}

/// Standard LayerNorm on the last dimension.
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let x = x.broadcast_div(&std)?;
    let x = x.broadcast_mul(weight)?;
    Ok(x.broadcast_add(bias)?)
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Compute mean and std over context, return (normalized, mean, std).
fn std_scale(x: &[f32]) -> (Vec<f32>, f32, f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    // minimum_scale = 1e-5 (matches Python TinyTimeMixerStdScaler)
    let std = (var + 1e-5).sqrt() as f32;
    let scaled: Vec<f32> = x.iter().map(|&v| (v as f32 - mean as f32) / std).collect();
    (scaled, mean as f32, std)
}

/// Extract `num_patches` patches of length `patch_length` with stride `patch_stride`.
/// Uses the last `patch_length + patch_stride * (num_patches - 1)` timesteps.
/// Left-pads with the first value when the context is shorter than needed.
fn patchify(x: &[f32], patch_length: usize, patch_stride: usize, num_patches: usize) -> Vec<Vec<f32>> {
    let new_seq_len = patch_length + patch_stride * (num_patches - 1);
    let padded: Vec<f32> = if x.len() < new_seq_len {
        let pad_val = x.first().copied().unwrap_or(0.0);
        let mut v = vec![pad_val; new_seq_len - x.len()];
        v.extend_from_slice(x);
        v
    } else {
        x[x.len() - new_seq_len..].to_vec()
    };
    (0..num_patches)
        .map(|i| padded[i * patch_stride..i * patch_stride + patch_length].to_vec())
        .collect()
}
