// Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
//
// This product includes software developed at Datadog (https://www.datadoghq.com/)
// Copyright 2026 Datadog, Inc.

//! Toto-2 inference engine.
//!
//! Unit-scaling (μP) inference notes:
//! * `uu.Linear` → `F.linear(x, w, b) / sqrt(fan_in)`
//! * `uu.LinearReadout` → `F.linear(x, w, b) / fan_in`
//! * `U.silu` → `F.silu(x) * 1.766782948` (unit-scaled)
//! * τ-rule: `residual_split` is identity; `residual_add(h,skip,τ)` = `h*(τ/d) + skip*(1/d)` where `d=√(1+τ²)`
//! * `PerDimScale(q, w)` = `q * softplus(w) / log(2)` — the 0.52103 unit-scaling factors cancel

mod rope;

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;

use rope::RopeCache;

const SILU_SCALE: f64 = 1.766782948312328;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InferConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_groups: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub patch_size: usize,
    pub norm_eps: f64,
    pub layer_group_size: usize,
    pub num_variate_layers_per_group: usize,
    pub variate_layer_first: bool,
    pub use_xpos: bool,
    pub residual_mult: f64,
    pub residual_attn_ratio: f64,
    /// Run forward pass in F64 to match PyTorch numerical precision.
    /// Weights are cast to F64 at load time (~2× memory). Default: false.
    pub compute_f64: bool,
}

impl InferConfig {
    fn is_variate_layer(&self, idx: usize) -> bool {
        if self.variate_layer_first {
            idx % self.layer_group_size < self.num_variate_layers_per_group
        } else {
            idx % self.layer_group_size
                >= self.layer_group_size - self.num_variate_layers_per_group
        }
    }
    fn q_size(&self) -> usize { self.qk_dim * self.num_heads }
    fn k_size(&self) -> usize { self.qk_dim * self.num_groups }
    fn v_size(&self) -> usize { self.v_dim * self.num_groups }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

struct ResidualMlpWeights {
    l1_w: Tensor, l1_b: Tensor,
    l2_w: Tensor, l2_b: Tensor,
    skip_w: Tensor, skip_b: Tensor,
    tau: f64,
    is_output: bool, // true → use LinearReadout scale (1/fan_in) for l2 and skip
}

struct BlockWeights {
    attn_qkv_w: Tensor,
    attn_qkv_b: Option<Tensor>,
    attn_out_w: Tensor,
    attn_out_b: Option<Tensor>,
    attn_pds: Option<Tensor>, // [qk_dim] raw param
    attn_tau: f64,
    ffn_up_w: Tensor,
    ffn_down_w: Tensor,
    mlp_tau: f64,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct TotoModel {
    device: Device,
    pub config: InferConfig,
    rope: RopeCache,
    patch_proj: ResidualMlpWeights,
    blocks: Vec<BlockWeights>,
    output_head: ResidualMlpWeights,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
}

// ---------------------------------------------------------------------------
// GGUF loading helpers (free functions to avoid multiple &mut borrows)
// ---------------------------------------------------------------------------

fn load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let qt = content
        .tensor(reader, name, device)
        .with_context(|| format!("load tensor '{name}'"))?;
    Ok(qt.dequantize(device)?.to_dtype(dtype)?)
}

fn try_load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    match content.tensor(reader, name, device) {
        Ok(qt) => Ok(Some(qt.dequantize(device)?.to_dtype(dtype)?)),
        Err(_) => Ok(None),
    }
}

/// Compute attn_tau and mlp_tau for each layer using the transformer residual scaling rule.
///
/// This reproduces `uu.transformer_residual_scaling_rule` + the per-layer buffer assignment
/// from `SelfAttentionTransformerLayer.__init__` in the Python model.
fn compute_taus(num_layers: usize, residual_mult: f64, residual_attn_ratio: f64) -> (Vec<f64>, Vec<f64>) {
    let total_depth = 2 * num_layers;
    let alpha_mlp = residual_mult * (2.0 / (1.0 + residual_attn_ratio.powi(2))).sqrt();
    let alpha_attn = residual_attn_ratio * alpha_mlp;

    let tau = |index: usize| -> f64 {
        let n_attn = (index + 1) / 2;
        let n_mlp = index / 2;
        let num = if index % 2 == 0 { alpha_attn } else { alpha_mlp };
        let den = (total_depth as f64 / 2.0
            + n_attn as f64 * alpha_attn.powi(2)
            + n_mlp as f64 * alpha_mlp.powi(2))
        .sqrt();
        num / den
    };

    (0..num_layers).map(|i| (tau(2 * i), tau(2 * i + 1))).unzip()
}

impl TotoModel {
    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        let device = Device::Cpu;
        let dtype = if config.compute_f64 { DType::F64 } else { DType::F32 };
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .context("parse GGUF header")?;

        macro_rules! ld {
            ($name:expr) => { load_tensor(&content, &mut reader, $name, &device, dtype) };
        }
        macro_rules! tld {
            ($name:expr) => { try_load_tensor(&content, &mut reader, $name, &device, dtype) };
        }

        // --- patch_proj ---
        let patch_proj = ResidualMlpWeights {
            l1_w: ld!("patch_proj.linear1.weight")?,
            l1_b: ld!("patch_proj.linear1.bias")?,
            l2_w: ld!("patch_proj.linear2.weight")?,
            l2_b: ld!("patch_proj.linear2.bias")?,
            skip_w: ld!("patch_proj.skip_proj.weight")?,
            skip_b: ld!("patch_proj.skip_proj.bias")?,
            tau: 1.0,
            is_output: false,
        };

        // Tau values are deterministic given the config — compute rather than load
        // (avoids candle's inability to dequantize 0-dim tensors)
        let (attn_taus, mlp_taus) = compute_taus(
            config.num_layers,
            config.residual_mult,
            config.residual_attn_ratio,
        );

        // --- transformer blocks ---
        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            blocks.push(BlockWeights {
                attn_qkv_w: ld!(&format!("blk.{n}.attn_qkv.weight"))?,
                attn_qkv_b: tld!(&format!("blk.{n}.attn_qkv.bias"))?,
                attn_out_w: ld!(&format!("blk.{n}.attn_output.weight"))?,
                attn_out_b: tld!(&format!("blk.{n}.attn_output.bias"))?,
                attn_pds:   tld!(&format!("blk.{n}.attn_pds.weight"))?,
                attn_tau: attn_taus[n],
                ffn_up_w:   ld!(&format!("blk.{n}.ffn_up.weight"))?,
                ffn_down_w: {
                    let w = ld!(&format!("blk.{n}.ffn_down.weight"))?;
                    // Q8_0 converter stores this weight transposed so candle's innermost
                    // dim (d_model) is block-aligned. Detect by dim(0) == d_ff and flip back.
                    if w.dim(0)? != config.d_model {
                        w.t()?.contiguous()?
                    } else {
                        w
                    }
                },
                mlp_tau: mlp_taus[n],
            });
        }

        // --- output_head ---
        let output_head = ResidualMlpWeights {
            l1_w: ld!("output_head.linear1.weight")?,
            l1_b: ld!("output_head.linear1.bias")?,
            l2_w: ld!("output_head.linear2.weight")?,
            l2_b: ld!("output_head.linear2.bias")?,
            skip_w: ld!("output_head.skip_proj.weight")?,
            skip_b: ld!("output_head.skip_proj.bias")?,
            tau: 1.0,
            is_output: true,
        };

        let rope = RopeCache::new(config.qk_dim, 8192);
        Ok(Self { device, config, rope, patch_proj, blocks, output_head, causal_mask_cache: Mutex::new(HashMap::new()) })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Forecast from context.
    ///
    /// `target[n_var][ctx_len]` — observed values.
    /// `mask[n_var][ctx_len]` — true for valid observations.
    /// Returns `quantiles[9][n_var][prediction_length]`.
    ///
    /// Toto is a full-context model: the prediction region is appended to the
    /// context as masked (all-zero) patches before the forward pass.  The model
    /// attends over the full sequence and outputs quantiles for the future region.
    pub fn forecast(
        &self,
        target: &[Vec<f32>],
        mask: &[Vec<bool>],
        prediction_length: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let n_var = target.len();
        let ctx_len = target[0].len();
        let patch_size = self.config.patch_size;

        if ctx_len % patch_size != 0 {
            bail!("ctx_len ({ctx_len}) must be divisible by patch_size ({patch_size})");
        }
        let ctx_patches = ctx_len / patch_size;
        // Number of forecast patches needed (ceiling division)
        let fcst_patches = (prediction_length + patch_size - 1) / patch_size;
        let total_patches = ctx_patches + fcst_patches;

        // Causal patched std scaler on context only
        let mut locs = Vec::with_capacity(n_var);
        let mut scales = Vec::with_capacity(n_var);
        for v in 0..n_var {
            let (loc, scale) = causal_patched_std_scaler(&target[v], &mask[v], patch_size);
            locs.push(loc);
            scales.push(scale);
        }

        // Build full input: context patches (observed) + forecast patches (masked zeros)
        // Shape: [1, n_var, total_patches, 2*patch_size]
        let mut patch_data = vec![0.0f32; n_var * total_patches * 2 * patch_size];
        for v in 0..n_var {
            // Context patches
            for p in 0..ctx_patches {
                let base = v * total_patches * 2 * patch_size + p * 2 * patch_size;
                for i in 0..patch_size {
                    let t = p * patch_size + i;
                    let obs = mask[v][t];
                    let val = if obs {
                        (((target[v][t] - locs[v][t]) / scales[v][t]) as f64).asinh() as f32
                    } else {
                        0.0
                    };
                    patch_data[base + i] = val;
                    patch_data[base + patch_size + i] = if obs { 0.0 } else { 1.0 };
                }
            }
            // Forecast patches: all zeros, not observed (mask channel = 1)
            for p in ctx_patches..total_patches {
                let base = v * total_patches * 2 * patch_size + p * 2 * patch_size;
                for i in 0..patch_size {
                    patch_data[base + i] = 0.0;
                    patch_data[base + patch_size + i] = 1.0; // not observed
                }
            }
        }

        let dtype = if self.config.compute_f64 { DType::F64 } else { DType::F32 };
        let x = Tensor::from_vec(
            patch_data,
            (1usize, n_var, total_patches, 2 * patch_size),
            &self.device,
        )?.to_dtype(dtype)?;

        // patch_proj (InputResidualMLP)
        let x = self.forward_residual_mlp(&x, &self.patch_proj)?;

        // transformer
        let x = self.forward_transformer(x, n_var, total_patches)?;

        // output_head (OutputResidualMLP)
        let x = self.forward_residual_mlp(&x, &self.output_head)?;
        // x: [1, n_var, total_patches, patch_size * 9]

        // Next-patch predictor: output at position i predicts patch i+1.
        // Python: x_out[..., -(block+1):-1] where block=fcst_patches
        // = positions [ctx_patches-1, ctx_patches+fcst_patches-1)
        let x = x.narrow(2, ctx_patches - 1, fcst_patches)?;
        // x: [1, n_var, fcst_patches, patch_size * 9]
        let fcst_steps = fcst_patches * patch_size;

        // Unflatten → [1, n_var, fcst_patches, patch_size, 9]
        let out = x.reshape((1usize, n_var, fcst_patches, patch_size, 9))?;
        // Permute → [9, 1, n_var, fcst_patches, patch_size]
        let out = out.permute([4, 0, 1, 2, 3])?.contiguous()?;
        // Flatten → [9, n_var, fcst_steps]
        let out = out.reshape((9usize, n_var, fcst_steps))?;

        // Trim to exactly prediction_length (if fcst_steps > prediction_length)
        let out = if fcst_steps > prediction_length {
            out.narrow(2, 0, prediction_length)?
        } else {
            out
        };
        // Cast back to F32 for output (harmless if already F32)
        let out_data = out.to_dtype(DType::F32)?.to_vec3::<f32>()?; // [9][n_var][pred_len]

        // Denormalise: sinh(quantile) * scale + loc (using final-patch stats)
        let loc_final: Vec<f32> = (0..n_var).map(|v| locs[v][ctx_len - 1]).collect();
        let scale_final: Vec<f32> = (0..n_var).map(|v| scales[v][ctx_len - 1]).collect();

        let mut result = vec![vec![vec![0.0f32; prediction_length]; n_var]; 9];
        for q in 0..9 {
            for v in 0..n_var {
                for t in 0..prediction_length {
                    let raw = out_data[q][v][t] as f64;
                    result[q][v][t] = (raw.sinh() as f32) * scale_final[v] + loc_final[v];
                }
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // ResidualMLP forward
    // -----------------------------------------------------------------------

    fn forward_residual_mlp(&self, x: &Tensor, w: &ResidualMlpWeights) -> Result<Tensor> {
        // residual_split at inference = identity; both branches see x
        let h = uu_linear(x, &w.l1_w, Some(&w.l1_b))?;
        let h = uu_silu(&h)?;

        let h = if w.is_output {
            uu_linear_readout(&h, &w.l2_w, Some(&w.l2_b))?
        } else {
            uu_linear(&h, &w.l2_w, Some(&w.l2_b))?
        };

        let skip = if w.is_output {
            uu_linear_readout(x, &w.skip_w, Some(&w.skip_b))?
        } else {
            uu_linear(x, &w.skip_w, Some(&w.skip_b))?
        };

        residual_add(&h, &skip, w.tau)
    }

    // -----------------------------------------------------------------------
    // Transformer
    // -----------------------------------------------------------------------

    fn forward_transformer(&self, mut x: Tensor, n_var: usize, num_patches: usize) -> Result<Tensor> {
        for (idx, blk) in self.blocks.iter().enumerate() {
            x = if self.config.is_variate_layer(idx) {
                self.forward_variate_layer(x, blk, n_var, num_patches)?
            } else {
                self.forward_time_layer(x, blk, n_var, num_patches)?
            };
        }
        rms_norm(&x, self.config.norm_eps)
    }

    fn forward_time_layer(
        &self,
        x: Tensor,
        blk: &BlockWeights,
        n_var: usize,
        num_patches: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        // [1, n_var, num_patches, d_model] → [n_var, num_patches, d_model]
        let state = x.reshape((n_var, num_patches, cfg.d_model))?;

        let normed = rms_norm(&state, cfg.norm_eps)?;
        let seq_ids: Vec<u32> = (0..num_patches as u32).collect();
        let attn_out = self.forward_attention(&normed, blk, &seq_ids, /*is_variate=*/false)?;
        let state = residual_add(&attn_out, &state, blk.attn_tau)?;

        let normed = rms_norm(&state, cfg.norm_eps)?;
        let ffn_out = self.forward_ffn(&normed, blk)?;
        let state = residual_add(&ffn_out, &state, blk.mlp_tau)?;

        // Restore [1, n_var, num_patches, d_model]
        Ok(state.reshape((1usize, n_var, num_patches, cfg.d_model))?)
    }

    fn forward_variate_layer(
        &self,
        x: Tensor,
        blk: &BlockWeights,
        n_var: usize,
        num_patches: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        // [1, n_var, num_patches, d] → [num_patches, n_var, d]
        let state = x.permute([0, 2, 1, 3])?.contiguous()?.reshape((num_patches, n_var, cfg.d_model))?;

        let normed = rms_norm(&state, cfg.norm_eps)?;
        let attn_out = self.forward_attention(&normed, blk, &[], /*is_variate=*/true)?;
        let state = residual_add(&attn_out, &state, blk.attn_tau)?;

        let normed = rms_norm(&state, cfg.norm_eps)?;
        let ffn_out = self.forward_ffn(&normed, blk)?;
        let state = residual_add(&ffn_out, &state, blk.mlp_tau)?;

        // [num_patches, n_var, d] → [1, n_var, num_patches, d]
        Ok(state
            .reshape((1usize, num_patches, n_var, cfg.d_model))?
            .permute([0, 2, 1, 3])?
            .contiguous()?)
    }

    // -----------------------------------------------------------------------
    // Self-attention
    // -----------------------------------------------------------------------

    fn forward_attention(
        &self,
        state: &Tensor,
        blk: &BlockWeights,
        seq_ids: &[u32],
        is_variate: bool,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let (batch, seq, _d) = state.dims3()?;

        // Fused QKV — narrow then contiguous (narrow on last-dim creates non-contiguous views)
        let qkv = uu_linear(state, &blk.attn_qkv_w, blk.attn_qkv_b.as_ref())?;
        let q = qkv.narrow(D::Minus1, 0, cfg.q_size())?.contiguous()?;
        let k = qkv.narrow(D::Minus1, cfg.q_size(), cfg.k_size())?.contiguous()?;
        let v = qkv.narrow(D::Minus1, cfg.q_size() + cfg.k_size(), cfg.v_size())?.contiguous()?;

        // [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        let q = q.reshape((batch, seq, cfg.num_heads, cfg.qk_dim))?.permute([0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape((batch, seq, cfg.num_groups, cfg.qk_dim))?.permute([0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape((batch, seq, cfg.num_groups, cfg.v_dim))?.permute([0, 2, 1, 3])?.contiguous()?;

        // PerDimScale on Q
        let q = match blk.attn_pds.as_ref() {
            Some(pds_w) => apply_per_dim_scale(&q, pds_w)?,
            None => q,
        };

        // xPos-RoPE (time layers only)
        let (q, k) = if !is_variate && !seq_ids.is_empty() && cfg.use_xpos {
            let q = self.rope.apply(&q, seq_ids, 1.0, &self.device)?;
            let k = self.rope.apply(&k, seq_ids, -1.0, &self.device)?;
            (q, k)
        } else {
            (q, k)
        };

        // MuP scale: 1/qk_dim
        let scale = 1.0 / cfg.qk_dim as f64;
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;

        // Causal mask for time layers
        let scores = if !is_variate {
            self.apply_causal_mask(scores, seq)?
        } else {
            scores
        };

        let attn = ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        // [batch, heads, seq, v_dim] → [batch, seq, heads*v_dim]
        let out = out.permute([0, 2, 1, 3])?.contiguous()?.reshape((batch, seq, cfg.num_heads * cfg.v_dim))?;

        uu_linear(&out, &blk.attn_out_w, blk.attn_out_b.as_ref())
    }

    // -----------------------------------------------------------------------
    // SwiGLU FFN
    // -----------------------------------------------------------------------

    fn forward_ffn(&self, x: &Tensor, blk: &BlockWeights) -> Result<Tensor> {
        let fc1_out = uu_linear(x, &blk.ffn_up_w, None)?;
        let half = fc1_out.dim(D::Minus1)? / 2;
        let gate = fc1_out.narrow(D::Minus1, 0, half)?;
        let val = fc1_out.narrow(D::Minus1, half, half)?;
        // Standard F.silu (not unit-scaled) as in the Python FFN
        let activated = (gate * ops::silu(&val)?)?;
        uu_linear(&activated, &blk.ffn_down_w, None)
    }
}

// ---------------------------------------------------------------------------
// Unit-scaling ops
// ---------------------------------------------------------------------------

/// Unit-scaled linear for arbitrary batch shapes.
///
/// Flattens leading dims to 2D for matmul (candle doesn't broadcast 2D weights against nD inputs),
/// then restores the original batch shape.
fn uu_linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    linear_with_scale(x, w, b, 1.0 / (w.dim(1)? as f64).sqrt())
}

fn uu_linear_readout(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    linear_with_scale(x, w, b, 1.0 / w.dim(1)? as f64)
}

fn linear_with_scale(x: &Tensor, w: &Tensor, b: Option<&Tensor>, scale: f64) -> Result<Tensor> {
    let shape = x.dims().to_vec();
    let d_in = *shape.last().unwrap();
    let batch: usize = shape[..shape.len() - 1].iter().product();
    let d_out = w.dim(0)?;

    let x_flat = x.reshape((batch, d_in))?;
    let out_flat = x_flat.matmul(&w.t()?)?; // [batch, d_out]

    let mut out_shape = shape[..shape.len() - 1].to_vec();
    out_shape.push(d_out);
    let out = out_flat.reshape(out_shape)?;
    let out = if let Some(b) = b { out.broadcast_add(b)? } else { out };
    Ok((out * scale)?)
}

fn uu_silu(x: &Tensor) -> Result<Tensor> {
    Ok((ops::silu(x)? * SILU_SCALE)?)
}

fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x2 = x.sqr()?;
    let mean_x2 = x2.mean_keepdim(D::Minus1)?;
    let rms = mean_x2.affine(1.0, eps)?.sqrt()?;
    Ok(x.broadcast_div(&rms)?)
}

fn residual_add(h: &Tensor, skip: &Tensor, tau: f64) -> Result<Tensor> {
    let denom = (1.0 + tau * tau).sqrt();
    Ok(((h * (tau / denom))? + (skip * (1.0 / denom))?)?)
}

fn apply_per_dim_scale(q: &Tensor, pds_w: &Tensor) -> Result<Tensor> {
    // Python: q * F.softplus(pds_w) / log(2)
    // The 0.52103 unit-scaling factors cancel between uu.softplus and per_dim_scale.
    let sp = (pds_w.exp()? + 1.0)?.log()?; // F.softplus
    let log2 = std::f64::consts::LN_2;
    let r = (sp / log2)?;
    Ok(q.broadcast_mul(&r)?)
}

impl TotoModel {
    fn apply_causal_mask(&self, scores: Tensor, seq: usize) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.lock().unwrap();
        if !cache.contains_key(&seq) {
            let mut mask_data = vec![0.0f32; seq * seq];
            for i in 0..seq {
                for j in (i + 1)..seq {
                    mask_data[i * seq + j] = f32::NEG_INFINITY;
                }
            }
            let dtype = if self.config.compute_f64 { DType::F64 } else { DType::F32 };
            let mask = Tensor::from_vec(mask_data, (seq, seq), &self.device)?.to_dtype(dtype)?;
            cache.insert(seq, mask);
        }
        Ok(scores.broadcast_add(&cache[&seq])?)
    }
}

// ---------------------------------------------------------------------------
// Causal patched std scaler
// ---------------------------------------------------------------------------

fn causal_patched_std_scaler(
    data: &[f32],
    mask: &[bool],
    patch_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = data.len();
    let num_patches = n / patch_size;

    let mut cum_count = 0.0f64;
    let mut m1 = 0.0f64;
    let mut m2 = 0.0f64;
    let correction = 1.0f64;
    let minimum_scale = 1e-6f64;

    let mut patch_loc = vec![0.0f32; num_patches];
    let mut patch_scale = vec![1e-6f32; num_patches];

    for p in 0..num_patches {
        for i in 0..patch_size {
            let t = p * patch_size + i;
            if mask[t] {
                let x = data[t] as f64;
                cum_count += 1.0;
                let prev_m1 = m1;
                m1 += (x - m1) / cum_count;
                m2 += (x - prev_m1) * (x - m1);
            }
        }
        patch_loc[p] = m1 as f32;
        let denom = (cum_count - correction).max(1.0);
        patch_scale[p] = (m2 / denom).sqrt().max(minimum_scale) as f32;
    }

    let mut loc = vec![0.0f32; n];
    let mut scale = vec![1e-6f32; n];
    for p in 0..num_patches {
        for i in 0..patch_size {
            let t = p * patch_size + i;
            loc[t] = patch_loc[p];
            scale[t] = patch_scale[p];
        }
    }
    (loc, scale)
}
