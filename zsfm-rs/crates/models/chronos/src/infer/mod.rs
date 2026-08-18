//! Chronos-2 inference engine.
//!
//! Architecture: encoder-only with alternating TimeSelfAttention + GroupSelfAttention + FFN.
//! Key differences from standard T5:
//! * Standard Llama RoPE (`rotate_half = [-x[half:], x[:half]]`)
//! * T5-style RMSNorm (no bias, no mean subtraction)
//! * No attention scale (scale=1.0 in MHA)
//! * GroupSelfAttention for batch=1 reduces to position-wise v → o projection

mod rope;

use std::io::{BufReader, Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use rope::RopeCache;
use zsfm_nn::{linear, load_weight};

use crate::config::Chronos2Config;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InferConfig {
    d_model: usize,
    d_kv: usize,
    d_ff: usize,
    num_layers: usize,
    num_heads: usize,
    layer_norm_eps: f64,
    rope_theta: f64,
    patch_size: usize,
    patch_stride: usize,
    context_length: usize,
    quantiles: Vec<f32>,
    use_reg_token: bool,
    use_arcsinh: bool,
    time_encoding_scale: usize,
    dense_act_fn: String,
}

/// `Chronos2Config` already resolved every default (via serde) or failed to parse if a
/// genuinely required field was missing, so this mapping is infallible and needs no further
/// defaulting of its own.
impl From<&Chronos2Config> for InferConfig {
    fn from(c2: &Chronos2Config) -> Self {
        let cc = &c2.chronos_config;
        InferConfig {
            d_model: c2.d_model as usize,
            d_kv: c2.d_kv as usize,
            d_ff: c2.d_ff as usize,
            num_layers: c2.num_layers as usize,
            num_heads: c2.num_heads as usize,
            layer_norm_eps: c2.layer_norm_epsilon,
            rope_theta: c2.rope_theta,
            patch_size: cc.input_patch_size as usize,
            patch_stride: cc.input_patch_stride as usize,
            context_length: cc.context_length as usize,
            quantiles: cc.quantiles.clone(),
            use_reg_token: cc.use_reg_token,
            use_arcsinh: cc.use_arcsinh,
            time_encoding_scale: c2.time_encoding_scale() as usize,
            dense_act_fn: c2.dense_act_fn().to_string(),
        }
    }
}

impl InferConfig {
    fn inner_dim(&self) -> usize { self.num_heads * self.d_kv }

    pub fn patch_size(&self) -> usize { self.patch_size }
    pub fn patch_stride(&self) -> usize { self.patch_stride }
    pub fn context_length(&self) -> usize { self.context_length }
    pub fn quantiles(&self) -> &[f32] { &self.quantiles }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent constructor for [`ChronosModel`]: point it at a GGUF file and a config (from a parsed
/// `config.json` via [`config_from`](ChronosModelBuilder::config_from)), then call
/// [`build`](ChronosModelBuilder::build).
///
/// ```no_run
/// use zsfm_chronos::{Chronos2Config, ChronosModel};
///
/// # fn main() -> anyhow::Result<()> {
/// let c2 = Chronos2Config::from_json(&std::fs::read_to_string("config.json")?)?;
/// let model = ChronosModel::builder("chronos.gguf").config_from(&c2).build()?;
/// # Ok(()) }
/// ```
pub struct ChronosModelBuilder {
    gguf_path: PathBuf,
    config: Option<InferConfig>,
}

impl ChronosModelBuilder {
    fn new(gguf_path: impl Into<PathBuf>) -> Self {
        Self { gguf_path: gguf_path.into(), config: None }
    }

    pub fn config(mut self, config: InferConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn config_from(mut self, c2: &Chronos2Config) -> Self {
        self.config = Some(InferConfig::from(c2));
        self
    }

    pub fn build(self) -> Result<ChronosModel> {
        let config = self
            .config
            .context("ChronosModelBuilder: no config set — call .config(...) or .config_from(...)")?;
        ChronosModel::load(&self.gguf_path, config)
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct ResidualBlockWeights {
    /// hidden_layer weight: (h_dim, in_dim) in PyTorch order
    hidden_w: Tensor,
    hidden_b: Tensor,
    /// output_layer weight: (out_dim, h_dim)
    output_w: Tensor,
    output_b: Tensor,
    /// residual_layer weight: (out_dim, in_dim)
    skip_w: Tensor,
    skip_b: Tensor,
}

struct AttnWeights {
    qkv_w: Tensor, // fused [3*inner_dim, d_model]
    o_w: Tensor,
    norm_w: Tensor,
}

struct FfnWeights {
    wi_w: Tensor,
    wo_w: Tensor,
    norm_w: Tensor,
}

struct BlockWeights {
    time_attn: AttnWeights,
    group_attn: AttnWeights,
    ffn: FfnWeights,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct ChronosModel {
    device: Device,
    pub config: InferConfig,
    rope: RopeCache,
    /// Token embedding (for [PAD]/[REG] special tokens).
    token_embd: Tensor,
    input_patch: ResidualBlockWeights,
    blocks: Vec<BlockWeights>,
    enc_norm: Tensor,
    output_patch: ResidualBlockWeights,
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
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    d_out_h: usize,  // hidden_layer d_out = h_dim
    d_out: usize,    // output_layer / skip d_out
    device: &Device,
) -> Result<ResidualBlockWeights> {
    // hidden_layer: (h_dim, in_dim)
    let hidden_w = load_weight(content, reader, &format!("{prefix}.hidden.weight"), d_out_h, device)?;
    let hidden_b = load_tensor(content, reader, &format!("{prefix}.hidden.bias"), device)?;
    // output_layer: (out_dim, h_dim)
    let output_w = load_weight(content, reader, &format!("{prefix}.output.weight"), d_out, device)?;
    let output_b = load_tensor(content, reader, &format!("{prefix}.output.bias"), device)?;
    // residual_layer: (out_dim, in_dim)
    let skip_w = load_weight(content, reader, &format!("{prefix}.skip.weight"), d_out, device)?;
    let skip_b = load_tensor(content, reader, &format!("{prefix}.skip.bias"), device)?;
    Ok(ResidualBlockWeights { hidden_w, hidden_b, output_w, output_b, skip_w, skip_b })
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    blk: usize,
    kind: &str,
    d_model: usize,
    inner_dim: usize,
    device: &Device,
) -> Result<AttnWeights> {
    let p = |s: &str| format!("blk.{blk}.{kind}.{s}");
    let norm_name = format!("blk.{blk}.{kind}_norm.weight");
    let q_w = load_weight(content, reader, &p("q.weight"), inner_dim, device)?;
    let k_w = load_weight(content, reader, &p("k.weight"), inner_dim, device)?;
    let v_w = load_weight(content, reader, &p("v.weight"), inner_dim, device)?;
    let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
        .with_context(|| format!("qkv cat blk.{blk}.{kind}"))?;
    Ok(AttnWeights {
        qkv_w,
        o_w:    load_weight(content, reader, &p("o.weight"), d_model, device)?,
        norm_w: load_tensor(content, reader, &norm_name, device)?,
    })
}

impl ChronosModel {
    /// Start building a [`ChronosModel`] — see [`ChronosModelBuilder`].
    pub fn builder(gguf_path: impl Into<PathBuf>) -> ChronosModelBuilder {
        ChronosModelBuilder::new(gguf_path)
    }

    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let d = config.d_model;
        let ff = config.d_ff;
        let id = config.inner_dim();
        let ps = config.patch_size;
        let nq = config.quantiles.len();

        let token_embd = load_tensor(&content, &mut reader, "token_embd.weight", &device)?;

        // input_patch_embedding: in=3*ps, h=d_ff, out=d_model
        let input_patch = load_residual_block(
            &content, &mut reader, "input_patch", ff, d, &device,
        )?;

        // Encoder blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            let time_attn = load_attn(&content, &mut reader, n, "time_attn", d, id, &device)?;
            let group_attn = load_attn(&content, &mut reader, n, "group_attn", d, id, &device)?;
            let ffn = FfnWeights {
                wi_w: load_weight(&content, &mut reader, &format!("blk.{n}.ffn.wi.weight"), ff, &device)?,
                wo_w: load_weight(&content, &mut reader, &format!("blk.{n}.ffn.wo.weight"), d, &device)?,
                norm_w: load_tensor(&content, &mut reader, &format!("blk.{n}.ffn_norm.weight"), &device)?,
            };
            blocks.push(BlockWeights { time_attn, group_attn, ffn });
        }

        let enc_norm = load_tensor(&content, &mut reader, "enc_norm.weight", &device)?;

        // output_patch_embedding: in=d_model, h=d_ff, out=num_quantiles*patch_size
        let out_d = nq * ps;
        let output_patch = load_residual_block(
            &content, &mut reader, "output_patch", ff, out_d, &device,
        )?;

        let rope = RopeCache::new(config.d_kv, 8192, config.rope_theta, &device)?;

        Ok(Self { device, config, rope, token_embd, input_patch, blocks, enc_norm, output_patch })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Forecast univariate time series.
    ///
    /// `context` is a slice of observed values. `prediction_length` is the
    /// number of future timesteps. Returns `quantiles[num_q][prediction_length]`.
    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let ps = cfg.patch_size;
        let stride = cfg.patch_stride;
        let n_quantiles = cfg.quantiles.len();

        // Number of output patches needed (ceiling division)
        let n_out_patches = (prediction_length + ps - 1) / ps;

        // --- 1. InstanceNorm ---
        let (normalized, loc, scale) = instance_norm(context, cfg.use_arcsinh);

        // --- 2. Patch context ---
        let padded = pad_for_patching(&normalized, ps);
        let n_ctx_patches = (padded.len() - ps) / stride + 1;

        // --- 3. Build input features for context patches [n_ctx_patches, 3*ps] ---
        let ctx_features = build_patch_features(
            &padded, n_ctx_patches, ps, stride,
            -((n_ctx_patches * ps) as f32), 0.0,  // time enc: [-n*ps, ..., -1] / scale
            cfg.time_encoding_scale as f32,
            true, // observed (mask = 1)
        );

        // --- 4. Build input features for future patches [n_out_patches, 3*ps] ---
        let fut_features = build_patch_features_future(
            n_out_patches, ps,
            0.0,  // start time index
            cfg.time_encoding_scale as f32,
        );

        // --- 5. Input patch embedding ---
        let ctx_tensor = Tensor::from_vec(ctx_features, (n_ctx_patches, 3 * ps), &self.device)?;
        let fut_tensor = Tensor::from_vec(fut_features, (n_out_patches, 3 * ps), &self.device)?;

        let ctx_embeds = self.forward_residual_block(&ctx_tensor, &self.input_patch)?;
        let fut_embeds = self.forward_residual_block(&fut_tensor, &self.input_patch)?;

        // --- 6. Optionally add [REG] token between context and future patches ---
        let seq = if cfg.use_reg_token {
            // [REG] token id is 1 (stored in shared embedding)
            let reg_embed = self.token_embd.narrow(0, 1, 1)?; // [1, d_model]
            Tensor::cat(&[&ctx_embeds, &reg_embed, &fut_embeds], 0)?
        } else {
            Tensor::cat(&[&ctx_embeds, &fut_embeds], 0)?
        };
        // seq: [total_seq, d_model]
        let total_seq = seq.dim(0)?;

        // Add batch dimension: [1, total_seq, d_model]
        let hidden = seq.unsqueeze(0)?;

        // --- 7. Encoder forward pass ---
        let hidden = self.forward_encoder(hidden, total_seq)?;
        // hidden: [1, total_seq, d_model]

        // --- 8. Slice last n_out_patches hidden states ---
        let forecast_embeds = hidden.narrow(1, total_seq - n_out_patches, n_out_patches)?;
        // [1, n_out_patches, d_model]

        // --- 9. Output patch embedding ---
        let forecast_embeds = forecast_embeds.squeeze(0)?; // [n_out_patches, d_model]
        let quantile_raw = self.forward_residual_block(&forecast_embeds, &self.output_patch)?;
        // [n_out_patches, num_quantiles * patch_size]

        // Reshape to [num_quantiles, n_out_patches * patch_size]
        let total_out = n_out_patches * ps;
        let quantile_raw = quantile_raw.reshape((n_out_patches, n_quantiles, ps))?;
        let quantile_raw = quantile_raw.permute([1, 0, 2])?.contiguous()?;
        let quantile_raw = quantile_raw.reshape((n_quantiles, total_out))?;

        // Trim to prediction_length
        let quantile_raw = if total_out > prediction_length {
            quantile_raw.narrow(1, 0, prediction_length)?
        } else {
            quantile_raw
        };

        // Unscale
        let data = quantile_raw.to_vec2::<f32>()?;
        let mut result = vec![vec![0.0f32; prediction_length]; n_quantiles];
        for q in 0..n_quantiles {
            for t in 0..prediction_length {
                let v = data[q][t] as f64;
                let v = if cfg.use_arcsinh { v.sinh() } else { v };
                result[q][t] = (v as f32) * scale + loc;
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Encoder
    // -----------------------------------------------------------------------

    fn forward_encoder(&self, mut x: Tensor, seq_len: usize) -> Result<Tensor> {
        for blk in &self.blocks {
            x = self.forward_block(x, blk, seq_len)?;
        }
        // Final layer norm: [1, seq_len, d_model]
        apply_t5_rms_norm(&x, &self.enc_norm, self.config.layer_norm_eps)
    }

    fn forward_block(
        &self,
        x: Tensor,
        blk: &BlockWeights,
        seq_len: usize,
    ) -> Result<Tensor> {
        // TimeSelfAttention: pre-norm + add residual
        let normed = apply_t5_rms_norm(&x, &blk.time_attn.norm_w, self.config.layer_norm_eps)?;
        let attn_out = self.forward_time_attn(&normed, &blk.time_attn, seq_len)?;
        let x = (&x + &attn_out)?;

        // GroupSelfAttention (batch=1 simplified: v + o projection)
        let normed = apply_t5_rms_norm(&x, &blk.group_attn.norm_w, self.config.layer_norm_eps)?;
        let grp_out = self.forward_group_attn_univariate(&normed, &blk.group_attn)?;
        let x = (&x + &grp_out)?;

        // FeedForward: pre-norm + add residual
        let normed = apply_t5_rms_norm(&x, &blk.ffn.norm_w, self.config.layer_norm_eps)?;
        let ffn_out = self.forward_ffn(&normed, &blk.ffn)?;
        Ok((&x + &ffn_out)?)
    }

    // -----------------------------------------------------------------------
    // TimeSelfAttention (bidirectional, with RoPE, scale=1.0)
    // -----------------------------------------------------------------------

    fn forward_time_attn(
        &self,
        x: &Tensor,        // [1, seq_len, d_model]
        w: &AttnWeights,
        seq_len: usize,
    ) -> Result<Tensor> {
        let id = self.config.inner_dim();
        let nh = self.config.num_heads;
        let dkv = self.config.d_kv;

        // Project q, k, v: single fused matmul → [1, seq_len, 3*inner_dim]
        let qkv = linear(x, &w.qkv_w, None)?;
        let q = qkv.narrow(D::Minus1, 0, id)?;
        let k = qkv.narrow(D::Minus1, id, id)?;
        let v = qkv.narrow(D::Minus1, 2 * id, id)?;

        // Reshape: [1, n_heads, seq_len, d_kv]
        let q = q.reshape((1, seq_len, nh, dkv))?.permute([0, 2, 1, 3])?.contiguous()?;
        let k = k.reshape((1, seq_len, nh, dkv))?.permute([0, 2, 1, 3])?.contiguous()?;
        let v = v.reshape((1, seq_len, nh, dkv))?.permute([0, 2, 1, 3])?.contiguous()?;

        let q = self.rope.apply(&q, seq_len)?;
        let k = self.rope.apply(&k, seq_len)?;

        // Attention: scores = q @ k^T (no scale)
        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        // Bidirectional: no mask — all zeros (all positions valid)
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        // Context: attn @ v → [1, n_heads, seq_len, d_kv]
        let out = attn.matmul(&v)?;

        // [1, n_heads, seq_len, d_kv] → [1, seq_len, inner_dim]
        let out = out.permute([0, 2, 1, 3])?.contiguous()?.reshape((1, seq_len, id))?;

        // Output projection
        linear(&out, &w.o_w, None)
    }

    // -----------------------------------------------------------------------
    // GroupSelfAttention — batch=1 (univariate) simplification.
    //
    // For a single time series the "batch" dimension has size 1, so each
    // timestep attends only to itself in the batch axis.  The attention
    // weights trivially become 1.0, reducing to:
    //   output = o_proj(v_proj(x))
    // -----------------------------------------------------------------------

    fn forward_group_attn_univariate(&self, x: &Tensor, w: &AttnWeights) -> Result<Tensor> {
        // x: [1, seq_len, d_model]  (after pre-norm, batch=1)
        // For univariate batch=1, group attention reduces to v_proj → o_proj.
        // Extract v slice from the fused qkv weight.
        let id = self.config.inner_dim();
        let qkv = linear(x, &w.qkv_w, None)?;
        let v = qkv.narrow(D::Minus1, 2 * id, id)?;
        linear(&v, &w.o_w, None)
    }

    // -----------------------------------------------------------------------
    // FeedForward: x → wi → act → wo  (no gating, default relu)
    // -----------------------------------------------------------------------

    fn forward_ffn(&self, x: &Tensor, w: &FfnWeights) -> Result<Tensor> {
        let h = linear(x, &w.wi_w, None)?;
        let h = apply_act(&h, &self.config.dense_act_fn)?;
        linear(&h, &w.wo_w, None)
    }

    // -----------------------------------------------------------------------
    // ResidualBlock: output_layer(act(hidden_layer(x))) + residual_layer(x)
    // -----------------------------------------------------------------------

    fn forward_residual_block(&self, x: &Tensor, w: &ResidualBlockWeights) -> Result<Tensor> {
        let h = linear(x, &w.hidden_w, Some(&w.hidden_b))?;
        let h = apply_act(&h, &self.config.dense_act_fn)?;
        let out = linear(&h, &w.output_w, Some(&w.output_b))?;
        let skip = linear(x, &w.skip_w, Some(&w.skip_b))?;
        Ok((&out + &skip)?)
    }
}

// ---------------------------------------------------------------------------
// Primitive ops
// ---------------------------------------------------------------------------

/// T5-style RMSNorm: weight * x * rsqrt(mean(x²) + eps).
/// No bias, no mean subtraction.
fn apply_t5_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    zsfm_nn::rms_norm(x, Some(weight), eps)
}

/// Apply dense activation function by name (supports "relu" and "gelu").
fn apply_act(x: &Tensor, name: &str) -> Result<Tensor> {
    match name {
        "relu" => Ok(x.relu()?),
        "gelu" => Ok(x.gelu_erf()?),       // standard erf-based gelu
        "gelu_new" | "gelu_pytorch_tanh" => Ok(x.gelu()?), // tanh approx
        "silu" | "swish" => Ok(candle_nn::ops::silu(x)?),
        other => bail!("unsupported activation function: {other}"),
    }
}

// ---------------------------------------------------------------------------
// InstanceNorm (standardization)
// ---------------------------------------------------------------------------

/// Subtract mean, divide by std, optionally apply arcsinh.
/// Returns (normalized, loc, scale) where loc and scale are scalars.
fn instance_norm(x: &[f32], use_arcsinh: bool) -> (Vec<f32>, f32, f32) {
    // Ignore NaN — for our use case input is fully observed.
    let n = x.len() as f64;
    let loc = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - loc).powi(2)).sum::<f64>() / n;
    let scale = (var.sqrt() as f32).max(1e-5);

    let mut out: Vec<f32> = x.iter().map(|&v| (v as f64 - loc) as f32 / scale).collect();
    if use_arcsinh {
        for v in &mut out {
            *v = (*v as f64).asinh() as f32;
        }
    }
    (out, loc as f32, scale)
}

// ---------------------------------------------------------------------------
// Patching
// ---------------------------------------------------------------------------

/// Pad the left side of the series with NaN so its length is divisible by patch_size.
fn pad_for_patching(x: &[f32], patch_size: usize) -> Vec<f32> {
    let rem = x.len() % patch_size;
    if rem == 0 {
        x.to_vec()
    } else {
        let pad_len = patch_size - rem;
        let mut out = vec![f32::NAN; pad_len];
        out.extend_from_slice(x);
        out
    }
}

/// Build concatenated [time_enc | values | mask] patches for context.
///
/// * time_enc: sequential indices from `time_start` advancing by 1 per timestep,
///   divided by `time_scale`.
/// * values: patch values (NaN positions → 0.0).
/// * mask: 1.0 where value is finite, 0.0 otherwise.
fn build_patch_features(
    padded: &[f32],
    n_patches: usize,
    patch_size: usize,
    stride: usize,
    time_start: f32,
    _time_end: f32,
    time_scale: f32,
    observed: bool,
) -> Vec<f32> {
    let feature_dim = 3 * patch_size;
    let mut out = vec![0.0f32; n_patches * feature_dim];

    for p in 0..n_patches {
        let offset = p * stride;
        let base = p * feature_dim;

        for i in 0..patch_size {
            let t = offset + i;
            let global_t = time_start + t as f32;

            // time encoding
            out[base + i] = global_t / time_scale;

            // value
            let v = padded.get(t).copied().unwrap_or(0.0);
            let is_obs = observed && v.is_finite();
            out[base + patch_size + i] = if is_obs { v } else { 0.0 };

            // mask: 1 if observed
            out[base + 2 * patch_size + i] = if is_obs { 1.0 } else { 0.0 };
        }
    }
    out
}

/// Build concatenated [time_enc | zeros | zeros] patches for future (no known values).
fn build_patch_features_future(
    n_patches: usize,
    patch_size: usize,
    time_start: f32,
    time_scale: f32,
) -> Vec<f32> {
    let feature_dim = 3 * patch_size;
    let mut out = vec![0.0f32; n_patches * feature_dim];

    for p in 0..n_patches {
        let offset = p * patch_size;
        let base = p * feature_dim;
        for i in 0..patch_size {
            let global_t = time_start + (offset + i) as f32;
            out[base + i] = global_t / time_scale;
            // values and mask remain 0.0
        }
    }
    out
}

// ---------------------------------------------------------------------------
// zsfm-core::Forecaster
// ---------------------------------------------------------------------------

impl zsfm_core::Forecaster for ChronosModel {
    type Config = InferConfig;

    fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        ChronosModel::load(gguf_path, config)
    }

    /// Chronos-2's `forecast()` is univariate-only; `context`/`mask` must carry exactly one
    /// variate (mask is currently unused — Chronos always treats the series as fully observed).
    fn forecast(
        &self,
        context: &[Vec<f32>],
        _mask: &[Vec<bool>],
        horizon: usize,
    ) -> Result<zsfm_core::QuantileMatrix> {
        anyhow::ensure!(context.len() == 1, "ChronosModel only supports univariate forecasting (1 variate)");
        let qmat = ChronosModel::forecast(self, &context[0], horizon)?; // [n_q][horizon]
        Ok(qmat.into_iter().map(|row| vec![row]).collect())
    }
}
