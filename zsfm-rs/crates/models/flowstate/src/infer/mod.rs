use std::collections::HashMap;
use std::io::BufReader;
use std::sync::Mutex;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::ops;
use simdeez::prelude::*;

use crate::config::FlowStateConfig;

simd_runtime_generate!(
    fn ssm_scan_step(
        state_r: &mut [f32], state_i: &mut [f32],
        a_r: &[f32], a_i: &[f32],
        bu_r: &[f32], bu_i: &[f32],
    ) {
        let mut sr = &mut state_r[..]; let mut si = &mut state_i[..];
        let mut ar = &a_r[..];         let mut ai = &a_i[..];
        let mut br = &bu_r[..];        let mut bi = &bu_i[..];

        while sr.len() >= S::Vf32::WIDTH {
            let sr_v = S::Vf32::load_from_slice(sr);
            let si_v = S::Vf32::load_from_slice(si);
            let ar_v = S::Vf32::load_from_slice(ar);
            let ai_v = S::Vf32::load_from_slice(ai);
            let br_v = S::Vf32::load_from_slice(br);
            let bi_v = S::Vf32::load_from_slice(bi);

            // new_r = ar*sr - ai*si + br  (neg_mul_add(a,b,c) = c - a*b)
            let new_r = ar_v.mul_add(sr_v, ai_v.neg_mul_add(si_v, br_v));
            // new_i = ar*si + ai*sr + bi
            let new_i = ar_v.mul_add(si_v, ai_v.mul_add(sr_v, bi_v));

            new_r.copy_to_slice(sr);
            new_i.copy_to_slice(si);

            sr = &mut sr[S::Vf32::WIDTH..]; si = &mut si[S::Vf32::WIDTH..];
            ar = &ar[S::Vf32::WIDTH..];     ai = &ai[S::Vf32::WIDTH..];
            br = &br[S::Vf32::WIDTH..];     bi = &bi[S::Vf32::WIDTH..];
        }

        for j in 0..sr.len() {
            let nr = ar[j] * sr[j] - ai[j] * si[j] + br[j];
            let ni = ar[j] * si[j] + ai[j] * sr[j] + bi[j];
            sr[j] = nr;
            si[j] = ni;
        }
    }
);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InferConfig {
    pub num_layers: usize,
    pub embed_dim: usize,
    pub state_dim: usize,
    pub n_inputs: usize,        // 2 for with_missing (value + mask)
    pub decoder_dim: usize,
    pub decoder_patch_len: usize,
    pub quantiles: Vec<f32>,
    pub basis_range: [f32; 2],  // e.g. [-1.0, 0.95] for "legs"
    pub context_length: usize,
    pub eps: f32,
}

/// `FlowStateConfig` already resolved every default (via serde) or failed to parse if a
/// genuinely required field was missing, so this mapping is infallible and needs no further
/// defaulting of its own.
impl From<&FlowStateConfig> for InferConfig {
    fn from(c: &FlowStateConfig) -> Self {
        InferConfig {
            num_layers:        c.encoder_num_layers as usize,
            embed_dim:         c.embedding_feature_dim as usize,
            state_dim:         c.encoder_state_dim as usize,
            n_inputs:          c.n_inputs() as usize,
            decoder_dim:       c.decoder_dim as usize,
            decoder_patch_len: c.decoder_patch_len as usize,
            quantiles:         c.quantiles.clone(),
            basis_range:       c.basis_range(),
            context_length:    c.context_length as usize,
            eps:               1e-5,
        }
    }
}

impl InferConfig {
    pub fn quantiles(&self) -> &[f32] { &self.quantiles }
    pub fn median_index(&self) -> usize {
        self.quantiles
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(self.quantiles.len() / 2)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent constructor for [`FlowStateModel`]: point it at a GGUF file and a config (from a
/// parsed `config.json` via [`config_from`](FlowStateModelBuilder::config_from)), then call
/// [`build`](FlowStateModelBuilder::build).
///
/// ```no_run
/// use zsfm_flowstate::{FlowStateConfig, FlowStateModel};
///
/// # fn main() -> anyhow::Result<()> {
/// let cfg = FlowStateConfig::from_json(&std::fs::read_to_string("config.json")?)?;
/// let model = FlowStateModel::builder("flowstate.gguf").config_from(&cfg).build()?;
/// # Ok(()) }
/// ```
pub struct FlowStateModelBuilder {
    gguf_path: PathBuf,
    config: Option<InferConfig>,
}

impl FlowStateModelBuilder {
    fn new(gguf_path: impl Into<PathBuf>) -> Self {
        Self { gguf_path: gguf_path.into(), config: None }
    }

    pub fn config(mut self, config: InferConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn config_from(mut self, c: &FlowStateConfig) -> Self {
        self.config = Some(InferConfig::from(c));
        self
    }

    pub fn build(self) -> Result<FlowStateModel> {
        let config = self
            .config
            .context("FlowStateModelBuilder: no config set — call .config(...) or .config_from(...)")?;
        FlowStateModel::load(&self.gguf_path, config)
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct S5Weights {
    log_lambda_real: Vec<f32>,  // [state_dim]
    lambda_imag:     Vec<f32>,  // [state_dim]
    b_r: Tensor,                // [state_dim, embed_dim]
    b_i: Tensor,                // [state_dim, embed_dim]
    c_r: Tensor,                // [embed_dim, state_dim]
    c_i: Tensor,                // [embed_dim, state_dim]
    d:   Tensor,                // [embed_dim]
    log_delta: Vec<f32>,        // [state_dim]
}

struct BlockWeights {
    ssm:        S5Weights,
    out_weight: Tensor,   // [embed_dim, embed_dim]
    out_bias:   Tensor,   // [embed_dim]
    norm_weight: Tensor,  // [embed_dim]
    norm_bias:   Tensor,  // [embed_dim]
    // cached per scale_factor: (A_bar_real, A_bar_imag, B_bar_real_t, B_bar_imag_t)
    disc_cache: Mutex<HashMap<u32, (Vec<f32>, Vec<f32>, Tensor, Tensor)>>,
}

pub struct FlowStateModel {
    device:     Device,
    pub config: InferConfig,
    embed_w:    Tensor,   // [embed_dim, n_inputs]
    embed_b:    Tensor,   // [embed_dim]
    blocks:     Vec<BlockWeights>,
    decoder_w:  Tensor,   // [n_quantiles * decoder_dim, embed_dim]
    decoder_b:  Tensor,   // [n_quantiles * decoder_dim]
    legendre_cache: Mutex<HashMap<usize, Tensor>>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn load_f32_vec(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
    device: &Device,
) -> anyhow::Result<Vec<f32>> {
    zsfm_nn::load_vec(content, reader, name, device)
}

fn load_matrix(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
    device: &Device,
) -> anyhow::Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

impl FlowStateModel {
    pub fn load(gguf_path: &Path, config: InferConfig) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut file = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut file).context("read GGUF header")?;

        let embed_w = load_matrix(&content, &mut file, "embed.weight", &device)?;
        let embed_b = load_matrix(&content, &mut file, "embed.bias", &device)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            let ssm = S5Weights {
                log_lambda_real: load_f32_vec(&content, &mut file, &format!("blk.{n}.ssm.log_lambda_real"), &device)?,
                lambda_imag:     load_f32_vec(&content, &mut file, &format!("blk.{n}.ssm.lambda_imag"),     &device)?,
                b_r: load_matrix(&content, &mut file, &format!("blk.{n}.ssm.b_r"), &device)?,
                b_i: load_matrix(&content, &mut file, &format!("blk.{n}.ssm.b_i"), &device)?,
                c_r: load_matrix(&content, &mut file, &format!("blk.{n}.ssm.c_r"), &device)?,
                c_i: load_matrix(&content, &mut file, &format!("blk.{n}.ssm.c_i"), &device)?,
                d:          load_matrix(&content, &mut file, &format!("blk.{n}.ssm.d"),         &device)?,
                log_delta:  load_f32_vec(&content, &mut file, &format!("blk.{n}.ssm.log_delta"), &device)?,
            };
            blocks.push(BlockWeights {
                ssm,
                out_weight:  load_matrix(&content, &mut file, &format!("blk.{n}.out.weight"),  &device)?,
                out_bias:    load_matrix(&content, &mut file, &format!("blk.{n}.out.bias"),    &device)?,
                norm_weight: load_matrix(&content, &mut file, &format!("blk.{n}.norm.weight"), &device)?,
                norm_bias:   load_matrix(&content, &mut file, &format!("blk.{n}.norm.bias"),   &device)?,
                disc_cache:  Mutex::new(HashMap::new()),
            });
        }

        let decoder_w = load_matrix(&content, &mut file, "decoder.weight", &device)?;
        let decoder_b = load_matrix(&content, &mut file, "decoder.bias", &device)?;

        Ok(Self {
            device,
            config,
            embed_w,
            embed_b,
            blocks,
            decoder_w,
            decoder_b,
            legendre_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Start building a [`FlowStateModel`] — see [`FlowStateModelBuilder`].
    pub fn builder(gguf_path: impl Into<PathBuf>) -> FlowStateModelBuilder {
        FlowStateModelBuilder::new(gguf_path)
    }

    // -----------------------------------------------------------------------
    // Public inference entry point
    // -----------------------------------------------------------------------

    /// Forecast `prediction_length` steps from a univariate context series.
    /// Returns `Vec<Vec<f32>>` of shape `[n_quantiles][prediction_length]`.
    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> anyhow::Result<Vec<Vec<f32>>> {
        let cfg = &self.config;

        // 1. Pad or trim context to match model requirements
        let ctx_len = context.len().min(cfg.context_length);
        let start = context.len().saturating_sub(ctx_len);
        let context = &context[start..];
        let seq_len = context.len();

        // 2. Causal RevIN: compute prefix statistics
        let (normed_values, final_mean, final_std) = causal_revin_norm(context, cfg.eps);
        if std::env::var("FLOWSTATE_DEBUG").is_ok() {
            eprintln!("RevIN final_mean={:.8} final_std={:.8}", final_mean, final_std);
            eprintln!("normed[0..4]: {:.6} {:.6} {:.6} {:.6}",
                      normed_values[0], normed_values[1], normed_values[2], normed_values[3]);
            eprintln!("normed[252..256]: {:.6} {:.6} {:.6} {:.6}",
                      normed_values[252], normed_values[253], normed_values[254], normed_values[255]);
        }

        // 3. Build input tensor [seq_len, n_inputs] with mask channel = 0 (no missing)
        let mut input_data = vec![0.0f32; seq_len * cfg.n_inputs];
        for t in 0..seq_len {
            input_data[t * cfg.n_inputs] = normed_values[t];
            if cfg.n_inputs > 1 {
                input_data[t * cfg.n_inputs + 1] = 0.0; // mask = 0 means known
            }
        }

        // 4. Embedding: [seq_len, n_inputs] × embed_w^T + embed_b → [seq_len, embed_dim]
        let input_t = Tensor::from_vec(input_data, (seq_len, cfg.n_inputs), &self.device)?;
        let mut hidden = linear(&input_t, &self.embed_w, &self.embed_b)?;

        // 5. Scale factor for discretization: decoder_patch_len / prediction_length
        let scale_factor = cfg.decoder_patch_len as f32 / prediction_length as f32;

        // 6. Encoder: S5 layers
        // Pre-allocate scratch buffers once and reuse across all blocks.
        // This avoids repeated large Vec allocations (seq_len * state_dim floats each)
        // that would otherwise be zero-initialised and discarded per non-last block.
        let scratch_len = seq_len * cfg.state_dim;
        let mut scan_r = vec![0.0f32; scratch_len];
        let mut scan_i = vec![0.0f32; scratch_len];
        let mut state_r = vec![0.0f32; cfg.state_dim];
        let mut state_i = vec![0.0f32; cfg.state_dim];
        let num_layers = self.blocks.len();
        for (i, block) in self.blocks.iter().enumerate() {
            let is_last = i == num_layers - 1;
            hidden = self.apply_s5_layer(
                hidden, block, scale_factor, is_last,
                &mut scan_r, &mut scan_i, &mut state_r, &mut state_i,
            )?;
        }
        // After last layer: hidden is [1, embed_dim]

        // 7. Decoder: linear → [n_q, decoder_dim]
        let n_q = cfg.quantiles.len();
        let coeffs = linear(&hidden, &self.decoder_w, &self.decoder_b)?
            .reshape((n_q, cfg.decoder_dim))?;

        if std::env::var("FLOWSTATE_DEBUG").is_ok() {
            let coeffs_data: Vec<f32> = coeffs.flatten_all()?.to_vec1()?;
            for qi in 0..n_q {
                for d in 0..cfg.decoder_dim {
                    eprintln!("COEFF,{qi},{d},{:.8}", coeffs_data[qi * cfg.decoder_dim + d]);
                }
            }
        }

        // 8. Legendre basis [prediction_length, decoder_dim] — cached per prediction_length
        let basis = {
            let mut cache = self.legendre_cache.lock().unwrap();
            if !cache.contains_key(&prediction_length) {
                let raw = legendre_basis(prediction_length, cfg.decoder_dim, cfg.basis_range,
                                        scale_factor, cfg.decoder_patch_len);
                let flat: Vec<f32> = raw.into_iter().flatten().collect();
                cache.insert(prediction_length,
                    Tensor::from_vec(flat, (prediction_length, cfg.decoder_dim), &self.device)?);
            }
            cache[&prediction_length].clone()
        };

        // 9. [n_q, decoder_dim] @ [decoder_dim, prediction_length] → [n_q, prediction_length]
        let out_t = coeffs.matmul(&basis.t()?)?;

        // 10. Denormalize raw decoder channels.
        let out_raw: Vec<f32> = out_t.flatten_all()?.to_vec1()?;
        let mut denormed = vec![vec![0.0f32; prediction_length]; n_q];
        for q in 0..n_q {
            for p in 0..prediction_length {
                denormed[q][p] = out_raw[q * prediction_length + p] * final_std + final_mean;
            }
        }

        // 11. Quantile recalibration: FlowStateForPrediction.forward() does not use the
        // n_q raw decoder channels directly as quantile predictions. It treats them as
        // n_q empirical samples and re-derives quantile estimates at the configured
        // probability levels via linear-interpolation order statistics
        // (`torch.quantile(model_output.last_hidden_state, quantiles, dim=1)` in
        // modeling_flowstate.py). Skipping this step caused outer quantiles (q0.1, q0.9)
        // to diverge from the Python reference by up to ~3.8 while q0.5 stayed near-exact
        // (interpolation index for p=0.5 lands exactly on the middle sorted sample).
        let mut output = vec![vec![0.0f32; prediction_length]; n_q];
        for p in 0..prediction_length {
            let mut sorted: Vec<f32> = (0..n_q).map(|q| denormed[q][p]).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (qi, &prob) in cfg.quantiles.iter().enumerate() {
                let idx = (n_q - 1) as f32 * prob;
                let lower = idx.floor() as usize;
                let upper = idx.ceil() as usize;
                let weight = idx - lower as f32;
                output[qi][p] = sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
            }
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // S5 layer (one encoder block)
    // -----------------------------------------------------------------------

    /// `scan_r` / `scan_i`: caller-owned scratch buffers of at least `seq_len * state_dim`
    /// elements, reused across blocks to avoid repeated large heap allocations.
    /// `state_r` / `state_i`: caller-owned scratch of at least `state_dim` elements.
    #[allow(clippy::too_many_arguments)]
    fn apply_s5_layer(
        &self,
        x: Tensor,          // [seq_len, embed_dim]
        block: &BlockWeights,
        scale_factor: f32,
        is_last: bool,
        scan_r: &mut Vec<f32>,  // scratch: seq_len * state_dim (reused across blocks)
        scan_i: &mut Vec<f32>,
        state_r: &mut Vec<f32>, // scratch: state_dim (running real state)
        state_i: &mut Vec<f32>, // scratch: state_dim (running imag state)
    ) -> anyhow::Result<Tensor> {
        let cfg = &self.config;
        let state_dim = cfg.state_dim;
        let embed_dim = cfg.embed_dim;

        let seq_len = x.dim(0)?;

        // Save skip connection (trimmed for last layer)
        let skip = if is_last {
            x.narrow(0, seq_len - 1, 1)?  // [1, embed_dim]
        } else {
            x.clone()
        };

        // ---- Get or compute discretized SSM matrices (cached per scale_factor) ----
        let (a_bar_r, a_bar_i, b_bar_r_t, b_bar_i_t) = {
            let key = scale_factor.to_bits();
            let mut cache = block.disc_cache.lock().unwrap();
            if !cache.contains_key(&key) {
                let result = discretize(&block.ssm, scale_factor, state_dim, embed_dim, &self.device)?;
                cache.insert(key, result);
            }
            let (ar, ai, brt, bit) = &cache[&key];
            (ar.clone(), ai.clone(), brt.clone(), bit.clone())
        };

        // B @ x for all timesteps at once: x [seq_len, embed_dim] × B^T [embed_dim, state_dim]
        let bu_r = x.matmul(&b_bar_r_t.t()?)?;  // [seq_len, state_dim]
        let bu_i = x.matmul(&b_bar_i_t.t()?)?;
        let bu_r_data: Vec<f32> = bu_r.flatten_all()?.to_vec1()?;
        let bu_i_data: Vec<f32> = bu_i.flatten_all()?.to_vec1()?;

        // Sequential SSM scan: h[t] = A_bar * h[t-1] + B_bar * u[t].
        // A_bar is diagonal complex, so each state dimension s is independent (NEON-vectorisable
        // inner loop). The outer loop over t is the inherent sequential dependency.
        // We reuse caller-provided scratch buffers to avoid repeated large heap allocations.
        //
        // Reset running state to zero at the start of each block.
        state_r[..state_dim].fill(0.0);
        state_i[..state_dim].fill(0.0);

        let (h_r_t, h_i_t) = if is_last {
            // Last block: only the final hidden state is consumed downstream.
            for t in 0..seq_len {
                let bu_r_t = &bu_r_data[t * state_dim..(t + 1) * state_dim];
                let bu_i_t = &bu_i_data[t * state_dim..(t + 1) * state_dim];
                ssm_scan_step(
                    &mut state_r[..state_dim], &mut state_i[..state_dim],
                    &a_bar_r, &a_bar_i, bu_r_t, bu_i_t,
                );
            }
            let hr = Tensor::from_vec(state_r[..state_dim].to_vec(), (1, state_dim), &self.device)?;
            let hi = Tensor::from_vec(state_i[..state_dim].to_vec(), (1, state_dim), &self.device)?;
            (hr, hi)
        } else {
            // Non-last blocks: full hidden state history required for the C projection.
            // Ensure scratch buffers are large enough (they are since caller sized them for
            // the maximum seq_len * state_dim of the first call in this forecast).
            let needed = seq_len * state_dim;
            if scan_r.len() < needed { scan_r.resize(needed, 0.0); }
            if scan_i.len() < needed { scan_i.resize(needed, 0.0); }
            for t in 0..seq_len {
                let bu_r_t = &bu_r_data[t * state_dim..(t + 1) * state_dim];
                let bu_i_t = &bu_i_data[t * state_dim..(t + 1) * state_dim];
                ssm_scan_step(
                    &mut state_r[..state_dim], &mut state_i[..state_dim],
                    &a_bar_r, &a_bar_i, bu_r_t, bu_i_t,
                );
                let row = t * state_dim;
                scan_r[row..row + state_dim].copy_from_slice(&state_r[..state_dim]);
                scan_i[row..row + state_dim].copy_from_slice(&state_i[..state_dim]);
            }
            let hr = Tensor::from_vec(scan_r[..needed].to_vec(), (seq_len, state_dim), &self.device)?;
            let hi = Tensor::from_vec(scan_i[..needed].to_vec(), (seq_len, state_dim), &self.device)?;
            (hr, hi)
        };

        // C @ h: y_real = C_r @ h_r - C_i @ h_i  → [out_seq_len, embed_dim]
        let y_from_cr = h_r_t.matmul(&block.ssm.c_r.t()?)?;
        let y_from_ci = h_i_t.matmul(&block.ssm.c_i.t()?)?;
        let y_raw = (y_from_cr - y_from_ci)?;

        // D skip: y += D * x_at_positions  (skip is the correct slice in both cases)
        let y_t = (y_raw + skip.broadcast_mul(&block.ssm.d)?)?;

        // ---- MLP: selu(y) * sigmoid(out_linear(selu(y))) ----
        let y_selu = selu_tensor(&y_t)?;
        let gate_pre = linear(&y_selu, &block.out_weight, &block.out_bias)?;
        let gate = sigmoid_tensor(&gate_pre)?;
        let y_gated = y_selu.mul(&gate)?;

        // ---- LayerNorm ----
        let y_normed = layer_norm(&y_gated, &block.norm_weight, &block.norm_bias, self.config.eps)?;

        // ---- Residual ----
        Ok((y_normed + skip)?)
    }
}

// ---------------------------------------------------------------------------
// Causal RevIN
// ---------------------------------------------------------------------------

/// Returns (normalized_values[seq_len], final_mean, final_std).
/// Each position t is normalized by the cumulative mean/std of x[0..=t].
fn causal_revin_norm(x: &[f32], eps: f32) -> (Vec<f32>, f32, f32) {
    let n = x.len();
    let mut normed = vec![0.0f32; n];
    let mut cum_sum = 0.0f32;
    let mut cum_sq_diff = 0.0f32;
    let mut final_mean = 0.0f32;
    let mut final_std = 1.0f32;

    for t in 0..n {
        let count = (t + 1) as f32;
        cum_sum += x[t];
        let mean_t = cum_sum / count;

        cum_sq_diff += (x[t] - mean_t) * (x[t] - mean_t);
        let var_t = (cum_sq_diff / count).max(0.0);
        let std_t = (var_t + eps).sqrt();

        normed[t] = (x[t] - mean_t) / std_t;

        if t == n - 1 {
            final_mean = mean_t;
            final_std = std_t;
        }
    }

    (normed, final_mean, final_std)
}

// ---------------------------------------------------------------------------
// SSM discretization
// ---------------------------------------------------------------------------

/// Returns (A_bar_real, A_bar_imag, B_bar_real_tensor, B_bar_imag_tensor).
/// B_bar tensors have shape [state_dim, embed_dim].
fn discretize(
    ssm: &S5Weights,
    scale_factor: f32,
    state_dim: usize,
    embed_dim: usize,
    device: &Device,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Tensor, Tensor)> {
    let mut a_r = vec![0.0f32; state_dim];
    let mut a_i = vec![0.0f32; state_dim];
    let mut coeff_r = vec![0.0f32; state_dim];
    let mut coeff_i = vec![0.0f32; state_dim];

    for s in 0..state_dim {
        let lam_r = -ssm.log_lambda_real[s].exp();
        let lam_i = ssm.lambda_imag[s];
        // Delta_eff = scale_factor * exp(log_Delta), matching modeling_flowstate.py's
        // `log_Lambda_bar = scale_factor * lambda_ * exp(log_Delta)`. NOT
        // exp(scale_factor * log_Delta) — the two coincide only at scale_factor == 1.0
        // (i.e. when horizon == decoder_patch_len), which masked this bug for the
        // common single-patch case.
        let delta = scale_factor * ssm.log_delta[s].exp();

        let exp_r = lam_r * delta;
        let exp_i = lam_i * delta;
        let mag = exp_r.exp();
        a_r[s] = mag * exp_i.cos();
        a_i[s] = mag * exp_i.sin();

        let num_r = a_r[s] - 1.0;
        let num_i = a_i[s];
        let denom_sq = lam_r * lam_r + lam_i * lam_i;
        if denom_sq > 1e-20 {
            coeff_r[s] = (num_r * lam_r + num_i * lam_i) / denom_sq;
            coeff_i[s] = (num_i * lam_r - num_r * lam_i) / denom_sq;
        } else {
            coeff_r[s] = delta;
            coeff_i[s] = 0.0;
        }
    }

    let b_r_data = get_tensor_data_row_major(&ssm.b_r, state_dim, embed_dim);
    let b_i_data = get_tensor_data_row_major(&ssm.b_i, state_dim, embed_dim);

    let mut b_bar_r = vec![0.0f32; state_dim * embed_dim];
    let mut b_bar_i = vec![0.0f32; state_dim * embed_dim];

    for s in 0..state_dim {
        for e in 0..embed_dim {
            let br = b_r_data[s * embed_dim + e];
            let bi = b_i_data[s * embed_dim + e];
            b_bar_r[s * embed_dim + e] = coeff_r[s] * br - coeff_i[s] * bi;
            b_bar_i[s * embed_dim + e] = coeff_r[s] * bi + coeff_i[s] * br;
        }
    }

    let b_bar_r_t = Tensor::from_vec(b_bar_r, (state_dim, embed_dim), device)?;
    let b_bar_i_t = Tensor::from_vec(b_bar_i, (state_dim, embed_dim), device)?;

    Ok((a_r, a_i, b_bar_r_t, b_bar_i_t))
}

/// Extract tensor data in row-major order as Vec<f32>.
fn get_tensor_data_row_major(t: &Tensor, rows: usize, cols: usize) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.reshape((rows, cols)))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1())
        .unwrap_or_else(|_| vec![0.0f32; rows * cols])
}

// ---------------------------------------------------------------------------
// Legendre basis (FlowStateLegendreBasis equivalent)
// ---------------------------------------------------------------------------

/// Public wrapper for diagnostics.
pub fn dump_legendre_basis(n_points: usize, degree: usize, range: [f32; 2],
                           scale: f32, pred_dist: usize) -> Vec<Vec<f32>> {
    legendre_basis(n_points, degree, range, scale, pred_dist)
}

/// Compute Legendre polynomial basis matrix.
/// Returns [n_points][degree+1] scaled by 1/4 (as in get_kernel).
fn legendre_basis(n_points: usize, degree: usize, range: [f32; 2],
                  scale: f32, pred_dist: usize) -> Vec<Vec<f32>> {
    let dt = scale * (range[1] - range[0]) / pred_dist as f32;
    let t: Vec<f32> = (1..=n_points).map(|i| range[0] + i as f32 * dt).collect();

    // Compute degree Legendre polynomials (P0..P_{degree-1}) per point.
    // degree+1 scratch columns needed during recurrence, then truncated to degree.
    let mut basis = vec![vec![0.0f32; degree + 1]; n_points];
    for (p, &x) in t.iter().enumerate() {
        basis[p][0] = 1.0;
        if degree >= 1 {
            basis[p][1] = x;
        }
        for k in 1..degree {
            let kf = k as f32;
            basis[p][k + 1] =
                ((2.0 * kf + 1.0) * x * basis[p][k] - kf * basis[p][k - 1]) / (kf + 1.0);
        }
        for d in 0..degree {
            basis[p][d] /= 4.0;
        }
        basis[p].truncate(degree);
    }

    basis
}

// ---------------------------------------------------------------------------
// Neural network primitives
// ---------------------------------------------------------------------------

/// y = x @ w^T + b  (w: [out, in], b: [out])
fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> anyhow::Result<Tensor> {
    zsfm_nn::linear_bias(x, w, b)
}

/// SELU activation using candle ops (no Vec roundtrip).
fn selu_tensor(x: &Tensor) -> anyhow::Result<Tensor> {
    const SCALE: f64 = 1.0507009873554804934193349852946;
    const ALPHA: f64 = 1.6732632423543772848170429916717;
    const ALPHA_SCALE: f64 = SCALE * ALPHA;
    let pos = x.relu()?;
    let neg = (x - &pos)?;            // min(x, 0)
    let selu_pos = (pos * SCALE)?;
    let selu_neg = ((neg.exp()? - 1.0)? * ALPHA_SCALE)?;
    Ok((selu_pos + selu_neg)?)
}

fn sigmoid_tensor(x: &Tensor) -> anyhow::Result<Tensor> {
    Ok(ops::sigmoid(x)?)
}

/// LayerNorm using candle ops (no Vec roundtrip).
fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> anyhow::Result<Tensor> {
    zsfm_nn::layer_norm(x, weight, bias, eps as f64)
}

// ---------------------------------------------------------------------------
// zsfm-core::Forecaster
// ---------------------------------------------------------------------------

impl zsfm_core::Forecaster for FlowStateModel {
    type Config = InferConfig;

    fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        FlowStateModel::load(gguf_path, config)
    }

    /// FlowState's `forecast()` is univariate-only; `context`/`mask` must carry exactly one
    /// variate (mask is currently unused — the missing-value channel is always fed as
    /// "known" since callers don't currently thread the mask through).
    fn forecast(
        &self,
        context: &[Vec<f32>],
        _mask: &[Vec<bool>],
        horizon: usize,
    ) -> Result<zsfm_core::QuantileMatrix> {
        anyhow::ensure!(context.len() == 1, "FlowStateModel only supports univariate forecasting (1 variate)");
        let qmat = FlowStateModel::forecast(self, &context[0], horizon)?; // [n_q][horizon]
        Ok(qmat.into_iter().map(|row| vec![row]).collect())
    }
}
