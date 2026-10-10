//! FlowState-R1 inference on the Burn backend (Flex CPU) for tensor
//! stages, with the original verbatim host kernels for the SSM core.
//!
//! Bit-exact port strategy: projections (embed, B/C, decoder, basis),
//! activations and norms run on Burn tensors; the diagonal SSM scan
//! (`ssm_scan_step`), discretization math, RevIN, Legendre basis and
//! quantile recalibration reuse the exact host code from `super` (copied
//! verbatim — it never touched candle). Weight loading goes through the
//! existing candle GGUF reader and converts each F32 tensor to Burn
//! `TensorData`, so there is exactly one GGUF parser.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use simdeez::prelude::*;
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_layer_norm_nd;
use zsfm_burn::norm::burn_rms_norm_nd;

use super::InferConfig;

fn device() -> Device {
    Device::flex()
}

fn burn2(t: &candle_core::Tensor) -> Result<Tensor<2>> {
    zsfm_burn::burn_weight_2d(t, &device())
}

fn burn1(t: &candle_core::Tensor) -> Result<Tensor<1>> {
    zsfm_burn::burn_weight_1d(t, &device())
}

fn load_t(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    burn2(&t)
}

fn load_v(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<1>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    burn1(&t)
}

fn load_f32_vec(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Vec<f32>> {
    Ok(zsfm_nn::load_vec(content, reader, name, &CDevice::Cpu)?)
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn layer_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, b: Tensor<1>, eps: f32) -> Tensor<D> {
    burn_layer_norm_nd(x, w, b, eps)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>, eps: f64) -> Tensor<D> {
    burn_rms_norm_nd(x, w, eps as f32)
}

fn selu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    const SCALE: f32 = 1.0507009873554804934193349852946;
    const ALPHA_SCALE: f32 = 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717;
    let pos = activation::relu(x.clone());
    let neg = x - pos.clone();
    let selu_pos = pos * SCALE;
    let selu_neg = (neg.exp() - 1.0) * ALPHA_SCALE;
    selu_pos + selu_neg
}

fn sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::sigmoid(x)
}

simd_runtime_generate!(
    fn ssm_scan_step(
        state_r: &mut [f32],
        state_i: &mut [f32],
        a_r: &[f32],
        a_i: &[f32],
        bu_r: &[f32],
        bu_i: &[f32],
    ) {
        let mut sr = &mut state_r[..];
        let mut si = &mut state_i[..];
        let mut ar = &a_r[..];
        let mut ai = &a_i[..];
        let mut br = &bu_r[..];
        let mut bi = &bu_i[..];

        while sr.len() >= S::Vf32::WIDTH {
            let sr_v = S::Vf32::load_from_slice(sr);
            let si_v = S::Vf32::load_from_slice(si);
            let ar_v = S::Vf32::load_from_slice(ar);
            let ai_v = S::Vf32::load_from_slice(ai);
            let br_v = S::Vf32::load_from_slice(br);
            let bi_v = S::Vf32::load_from_slice(bi);

            let new_r = ar_v.mul_add(sr_v, ai_v.neg_mul_add(si_v, br_v));
            let new_i = ar_v.mul_add(si_v, ai_v.mul_add(sr_v, bi_v));

            new_r.copy_to_slice(sr);
            new_i.copy_to_slice(si);

            sr = &mut sr[S::Vf32::WIDTH..];
            si = &mut si[S::Vf32::WIDTH..];
            ar = &ar[S::Vf32::WIDTH..];
            ai = &ai[S::Vf32::WIDTH..];
            br = &br[S::Vf32::WIDTH..];
            bi = &bi[S::Vf32::WIDTH..];
        }

        for j in 0..sr.len() {
            let nr = ar[j] * sr[j] - ai[j] * si[j] + br[j];
            let ni = ar[j] * si[j] + ai[j] * sr[j] + bi[j];
            sr[j] = nr;
            si[j] = ni;
        }
    }
);

struct BurnS5W {
    log_lambda_real: Vec<f32>,
    lambda_imag: Vec<f32>,
    b_r: Tensor<2>,
    b_i: Tensor<2>,
    c_r: Tensor<2>,
    c_i: Tensor<2>,
    d: Tensor<1>,
    log_delta: Vec<f32>,
}

struct BurnBlockW {
    ssm: BurnS5W,
    out_weight: Tensor<2>,
    out_bias: Tensor<1>,
    norm_weight: Tensor<1>,
    norm_bias: Tensor<1>,
    disc_cache: Mutex<HashMap<u32, (Vec<f32>, Vec<f32>, Tensor<2>, Tensor<2>)>>,
}

pub struct BurnFlowStateModel {
    config: InferConfig,
    embed_w: Tensor<2>,
    embed_b: Tensor<1>,
    blocks: Vec<BurnBlockW>,
    decoder_w: Tensor<2>,
    decoder_b: Tensor<1>,
    legendre_cache: Mutex<HashMap<usize, Tensor<2>>>,
}

impl BurnFlowStateModel {
    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let embed_w = load_t(&content, &mut reader, "embed.weight")?;
        let embed_b = load_v(&content, &mut reader, "embed.bias")?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            blocks.push(BurnBlockW {
                ssm: BurnS5W {
                    log_lambda_real: load_f32_vec(
                        &content,
                        &mut reader,
                        &format!("blk.{n}.ssm.log_lambda_real"),
                    )?,
                    lambda_imag: load_f32_vec(
                        &content,
                        &mut reader,
                        &format!("blk.{n}.ssm.lambda_imag"),
                    )?,
                    b_r: load_t(&content, &mut reader, &format!("blk.{n}.ssm.b_r"))?,
                    b_i: load_t(&content, &mut reader, &format!("blk.{n}.ssm.b_i"))?,
                    c_r: load_t(&content, &mut reader, &format!("blk.{n}.ssm.c_r"))?,
                    c_i: load_t(&content, &mut reader, &format!("blk.{n}.ssm.c_i"))?,
                    d: load_v(&content, &mut reader, &format!("blk.{n}.ssm.d"))?,
                    log_delta: load_f32_vec(
                        &content,
                        &mut reader,
                        &format!("blk.{n}.ssm.log_delta"),
                    )?,
                },
                out_weight: load_t(&content, &mut reader, &format!("blk.{n}.out.weight"))?,
                out_bias: load_v(&content, &mut reader, &format!("blk.{n}.out.bias"))?,
                norm_weight: load_v(&content, &mut reader, &format!("blk.{n}.norm.weight"))?,
                norm_bias: load_v(&content, &mut reader, &format!("blk.{n}.norm.bias"))?,
                disc_cache: Mutex::new(HashMap::new()),
            });
        }
        Ok(Self {
            config,
            embed_w,
            embed_b,
            blocks,
            decoder_w: load_t(&content, &mut reader, "decoder.weight")?,
            decoder_b: load_v(&content, &mut reader, "decoder.bias")?,
            legendre_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn quantiles(&self) -> &[f32] {
        self.config.quantiles()
    }

    pub fn median_index(&self) -> usize {
        self.config.median_index()
    }

    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let ctx_len = context.len().min(cfg.context_length);
        let start = context.len().saturating_sub(ctx_len);
        let context = &context[start..];
        let seq_len = context.len();
        let (normed_values, final_mean, final_std) = causal_revin_norm(context, cfg.eps);
        let mut input_data = vec![0.0f32; seq_len * cfg.n_inputs];
        for t in 0..seq_len {
            input_data[t * cfg.n_inputs] = normed_values[t];
            if cfg.n_inputs > 1 {
                input_data[t * cfg.n_inputs + 1] = 0.0;
            }
        }
        let dev = device();
        let input_t =
            Tensor::<2>::from_data(TensorData::new(input_data, [seq_len, cfg.n_inputs]), &dev);
        let mut hidden = linear(input_t, self.embed_w.clone(), Some(self.embed_b.clone()));
        let scale_factor = cfg.decoder_patch_len as f32 / prediction_length as f32;
        let scratch_len = seq_len * cfg.state_dim;
        let mut scan_r = vec![0.0f32; scratch_len];
        let mut scan_i = vec![0.0f32; scratch_len];
        let mut state_r = vec![0.0f32; cfg.state_dim];
        let mut state_i = vec![0.0f32; cfg.state_dim];
        let num_layers = self.blocks.len();
        for (i, block) in self.blocks.iter().enumerate() {
            let is_last = i == num_layers - 1;
            hidden = self.apply_s5_layer(
                hidden,
                block,
                scale_factor,
                is_last,
                &mut scan_r,
                &mut scan_i,
                &mut state_r,
                &mut state_i,
            )?;
        }
        let n_q = cfg.quantiles.len();
        let coeffs = linear(hidden, self.decoder_w.clone(), Some(self.decoder_b.clone()));
        let coeffs: Tensor<2> = coeffs.reshape([n_q, cfg.decoder_dim]);
        let basis = {
            let mut cache = self
                .legendre_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("cache mutex poisoned"))?;
            if !cache.contains_key(&prediction_length) {
                let raw = legendre_basis(
                    prediction_length,
                    cfg.decoder_dim,
                    cfg.basis_range,
                    scale_factor,
                    cfg.decoder_patch_len,
                );
                let flat: Vec<f32> = raw.into_iter().flatten().collect();
                cache.insert(
                    prediction_length,
                    Tensor::<2>::from_data(
                        TensorData::new(flat, [prediction_length, cfg.decoder_dim]),
                        &dev,
                    ),
                );
            }
            cache[&prediction_length].clone()
        };
        let out_t = coeffs.matmul(basis.transpose());
        let out_raw: Vec<f32> = to_host(&out_t)?;
        let mut denormed = vec![vec![0.0f32; prediction_length]; n_q];
        for q in 0..n_q {
            for p in 0..prediction_length {
                denormed[q][p] = out_raw[q * prediction_length + p] * final_std + final_mean;
            }
        }
        let mut output = vec![vec![0.0f32; prediction_length]; n_q];
        for p in 0..prediction_length {
            let mut sorted: Vec<f32> = (0..n_q).map(|q| denormed[q][p]).collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
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

    #[allow(clippy::too_many_arguments)]
    fn apply_s5_layer(
        &self,
        x: Tensor<2>,
        block: &BurnBlockW,
        scale_factor: f32,
        is_last: bool,
        scan_r: &mut Vec<f32>,
        scan_i: &mut Vec<f32>,
        state_r: &mut Vec<f32>,
        state_i: &mut Vec<f32>,
    ) -> Result<Tensor<2>> {
        let cfg = &self.config;
        let state_dim = cfg.state_dim;
        let embed_dim = cfg.embed_dim;
        let seq_len = x.dims()[0];
        let skip: Tensor<2> = if is_last {
            x.clone().narrow(0, seq_len - 1, 1)
        } else {
            x.clone()
        };
        let (a_bar_r, a_bar_i, b_bar_r, b_bar_i) = {
            let key = scale_factor.to_bits();
            let mut cache = block
                .disc_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("cache mutex poisoned"))?;
            if !cache.contains_key(&key) {
                let result = discretize(&block.ssm, scale_factor, state_dim, embed_dim)?;
                cache.insert(key, result);
            }
            let (ar, ai, brt, bit) = &cache[&key];
            (ar.clone(), ai.clone(), brt.clone(), bit.clone())
        };
        let bu_r = x.clone().matmul(b_bar_r.transpose());
        let bu_i = x.matmul(b_bar_i.transpose());
        let bu_r_data: Vec<f32> = to_host(&bu_r)?;
        let bu_i_data: Vec<f32> = to_host(&bu_i)?;
        state_r[..state_dim].fill(0.0);
        state_i[..state_dim].fill(0.0);
        let (h_r_t, h_i_t) = if is_last {
            for t in 0..seq_len {
                let bu_r_t = &bu_r_data[t * state_dim..(t + 1) * state_dim];
                let bu_i_t = &bu_i_data[t * state_dim..(t + 1) * state_dim];
                ssm_scan_step(
                    &mut state_r[..state_dim],
                    &mut state_i[..state_dim],
                    &a_bar_r,
                    &a_bar_i,
                    bu_r_t,
                    bu_i_t,
                );
            }
            let dev = device();
            (
                Tensor::<2>::from_data(
                    TensorData::new(state_r[..state_dim].to_vec(), [1, state_dim]),
                    &dev,
                ),
                Tensor::<2>::from_data(
                    TensorData::new(state_i[..state_dim].to_vec(), [1, state_dim]),
                    &dev,
                ),
            )
        } else {
            let needed = seq_len * state_dim;
            if scan_r.len() < needed {
                scan_r.resize(needed, 0.0);
            }
            if scan_i.len() < needed {
                scan_i.resize(needed, 0.0);
            }
            for t in 0..seq_len {
                let bu_r_t = &bu_r_data[t * state_dim..(t + 1) * state_dim];
                let bu_i_t = &bu_i_data[t * state_dim..(t + 1) * state_dim];
                ssm_scan_step(
                    &mut state_r[..state_dim],
                    &mut state_i[..state_dim],
                    &a_bar_r,
                    &a_bar_i,
                    bu_r_t,
                    bu_i_t,
                );
                let row = t * state_dim;
                scan_r[row..row + state_dim].copy_from_slice(&state_r[..state_dim]);
                scan_i[row..row + state_dim].copy_from_slice(&state_i[..state_dim]);
            }
            let dev = device();
            (
                Tensor::<2>::from_data(
                    TensorData::new(scan_r[..needed].to_vec(), [seq_len, state_dim]),
                    &dev,
                ),
                Tensor::<2>::from_data(
                    TensorData::new(scan_i[..needed].to_vec(), [seq_len, state_dim]),
                    &dev,
                ),
            )
        };
        let y_from_cr = h_r_t.matmul(block.ssm.c_r.clone().transpose());
        let y_from_ci = h_i_t.matmul(block.ssm.c_i.clone().transpose());
        let y_raw = y_from_cr - y_from_ci;
        let d: Tensor<1> = block.ssm.d.clone();
        let y_t = y_raw + skip.clone() * d.unsqueeze_dim::<2>(0);
        let y_selu = selu(y_t);
        let gate_pre = linear(
            y_selu.clone(),
            block.out_weight.clone(),
            Some(block.out_bias.clone()),
        );
        let gate = sigmoid(gate_pre);
        let y_gated = y_selu * gate;
        let y_normed = layer_norm(
            y_gated,
            block.norm_weight.clone(),
            block.norm_bias.clone(),
            self.config.eps,
        );
        Ok(y_normed + skip)
    }
}

fn discretize(
    ssm: &BurnS5W,
    scale_factor: f32,
    state_dim: usize,
    embed_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Tensor<2>, Tensor<2>)> {
    let mut a_r = vec![0.0f32; state_dim];
    let mut a_i = vec![0.0f32; state_dim];
    let mut coeff_r = vec![0.0f32; state_dim];
    let mut coeff_i = vec![0.0f32; state_dim];
    for s in 0..state_dim {
        let lam_r = -ssm.log_lambda_real[s].exp();
        let lam_i = ssm.lambda_imag[s];
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
    let b_r_data: Vec<f32> = to_host(&ssm.b_r)?;
    let b_i_data: Vec<f32> = to_host(&ssm.b_i)?;
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
    let dev = device();
    let b_bar_r_t = Tensor::<2>::from_data(TensorData::new(b_bar_r, [state_dim, embed_dim]), &dev);
    let b_bar_i_t = Tensor::<2>::from_data(TensorData::new(b_bar_i, [state_dim, embed_dim]), &dev);
    Ok((a_r, a_i, b_bar_r_t, b_bar_i_t))
}

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

fn legendre_basis(
    n_points: usize,
    degree: usize,
    range: [f32; 2],
    scale: f32,
    pred_dist: usize,
) -> Vec<Vec<f32>> {
    let dt = scale * (range[1] - range[0]) / pred_dist as f32;
    let t: Vec<f32> = (1..=n_points).map(|i| range[0] + i as f32 * dt).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use zsfm_gguf::{GGMLType, GGUFMetaValue, GGUFWriter};

    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn tiny_config() -> InferConfig {
        InferConfig {
            num_layers: 1,
            embed_dim: 8,
            state_dim: 8,
            n_inputs: 2,
            decoder_dim: 4,
            decoder_patch_len: 2,
            quantiles: vec![0.25, 0.5, 0.75],
            basis_range: [-1.0, 0.95],
            context_length: 16,
            eps: 1e-5,
        }
    }

    fn put2(w: &mut GGUFWriter, name: &str, rows: usize, cols: usize, seed: &mut u64, scale: f32) {
        *seed += 1;
        let data: Vec<f32> = pseudo(rows * cols, *seed)
            .iter()
            .map(|v| v * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(name, vec![cols as u64, rows as u64], GGMLType::F32, bytes);
    }

    fn put1(w: &mut GGUFWriter, name: &str, n: usize, seed: &mut u64) {
        *seed += 1;
        let bytes: Vec<u8> = pseudo(n, *seed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor(name, vec![n as u64], GGMLType::F32, bytes);
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        let dir = std::env::temp_dir().join(format!("flowstate-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("flowstate".into()),
        );
        let mut seed = 3100u64;
        let s = 0.05;
        put2(&mut w, "embed.weight", 8, 2, &mut seed, s);
        put1(&mut w, "embed.bias", 8, &mut seed);
        put1(&mut w, "blk.0.ssm.log_lambda_real", 8, &mut seed);
        put1(&mut w, "blk.0.ssm.lambda_imag", 8, &mut seed);
        put2(&mut w, "blk.0.ssm.b_r", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.ssm.b_i", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.ssm.c_r", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.ssm.c_i", 8, 8, &mut seed, s);
        put1(&mut w, "blk.0.ssm.d", 8, &mut seed);
        put1(&mut w, "blk.0.ssm.log_delta", 8, &mut seed);
        put2(&mut w, "blk.0.out.weight", 8, 8, &mut seed, s);
        put1(&mut w, "blk.0.out.bias", 8, &mut seed);
        put1(&mut w, "blk.0.norm.weight", 8, &mut seed);
        put1(&mut w, "blk.0.norm.bias", 8, &mut seed);
        put2(&mut w, "decoder.weight", 12, 8, &mut seed, s);
        put1(&mut w, "decoder.bias", 12, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::FlowStateModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnFlowStateModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, 2).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx, 2).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let mut err = 0.0f32;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                err = err.max((x - y).abs());
            }
        }
        println!("flowstate synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-3, "flowstate Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
