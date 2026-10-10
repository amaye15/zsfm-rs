//! TabDPT inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): outer impute/scale/pad, clip +
//! normalize, thinking rows, per-layer y-encoders, gated RMSNorm attention
//! with length-adaptive temperature, SwiGLU-split FFN, binned head. Weight
//! loading goes through the existing candle GGUF reader and converts each
//! F32 tensor to Burn `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::{burn_linear_bias, burn_linear_nobias};
use zsfm_burn::norm::{burn_layer_norm_nd, burn_rms_norm_nd};

use crate::config::TabDptConfig;

const LN_EPS: f32 = 1e-5;
const RMS_EPS: f32 = 1e-8;
const CLIP_N_SIGMA: f32 = 8.0;

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

fn linear(x: Tensor<2>, w: Tensor<2>, b: Tensor<1>) -> Tensor<2> {
    burn_linear_bias(x, w, b)
}

fn linear_o<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    zsfm_burn::linear::burn_linear_nd(x, w, b)
}

fn linear_nd<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    zsfm_burn::linear::burn_linear_nd(x, w, b)
}

fn layer_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, b: Tensor<1>) -> Tensor<D> {
    burn_layer_norm_nd(x, w, b, LN_EPS)
}

fn layer_norm_no_affine(x: Tensor<2>) -> Tensor<2> {
    zsfm_burn::norm::burn_layer_norm_no_affine(x, LN_EPS)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>) -> Tensor<D> {
    burn_rms_norm_nd(x, w, RMS_EPS)
}

fn gelu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::gelu(x)
}

fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::silu(x)
}

fn sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::sigmoid(x)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnBlockW {
    attn_norm_w: Tensor<1>,
    attn_norm_b: Tensor<1>,
    ff_norm_w: Tensor<1>,
    ff_norm_b: Tensor<1>,
    q_proj_w: Tensor<2>,
    k_proj_w: Tensor<2>,
    v_proj_w: Tensor<2>,
    out_proj_w: Tensor<2>,
    q_gate_w: Tensor<2>,
    q_norm_w: Tensor<1>,
    k_norm_w: Tensor<1>,
    ff_up_w: Tensor<2>,
    ff_down_w: Tensor<2>,
}

struct BurnYEncW {
    fc1_w: Tensor<2>,
    fc1_b: Tensor<1>,
    fc2_w: Tensor<2>,
    fc2_b: Tensor<1>,
}

pub struct BurnTabDptModel {
    config: TabDptConfig,
    encoder_w: Tensor<2>,
    encoder_b: Tensor<1>,
    thinking_embed: Tensor<2>,
    blocks: Vec<BurnBlockW>,
    y_encoders: Vec<BurnYEncW>,
    head_fc1_w: Tensor<2>,
    head_fc1_b: Tensor<1>,
    head_fc2_w: Tensor<2>,
    head_fc2_b: Tensor<1>,
}

impl BurnTabDptModel {
    pub fn load(gguf_path: &Path, config: TabDptConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let encoder_w = load_t(&content, &mut reader, "encoder.weight")?;
        let encoder_b = load_v(&content, &mut reader, "encoder.bias")?;
        let thinking_embed = load_t(&content, &mut reader, "thinking_embed")?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        let mut y_encoders = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            blocks.push(BurnBlockW {
                attn_norm_w: load_v(&content, &mut reader, &p("attn_norm.weight"))?,
                attn_norm_b: load_v(&content, &mut reader, &p("attn_norm.bias"))?,
                ff_norm_w: load_v(&content, &mut reader, &p("ff_norm.weight"))?,
                ff_norm_b: load_v(&content, &mut reader, &p("ff_norm.bias"))?,
                q_proj_w: load_t(&content, &mut reader, &p("q_proj.weight"))?,
                k_proj_w: load_t(&content, &mut reader, &p("k_proj.weight"))?,
                v_proj_w: load_t(&content, &mut reader, &p("v_proj.weight"))?,
                out_proj_w: load_t(&content, &mut reader, &p("out_proj.weight"))?,
                q_gate_w: load_t(&content, &mut reader, &p("q_gate.weight"))?,
                q_norm_w: load_v(&content, &mut reader, &p("q_norm.weight"))?,
                k_norm_w: load_v(&content, &mut reader, &p("k_norm.weight"))?,
                ff_up_w: load_t(&content, &mut reader, &p("ff_up.weight"))?,
                ff_down_w: load_t(&content, &mut reader, &p("ff_down.weight"))?,
            });
            let yp = |s: &str| format!("y_enc.{n}.{s}");
            y_encoders.push(BurnYEncW {
                fc1_w: load_t(&content, &mut reader, &yp("fc1.weight"))?,
                fc1_b: load_v(&content, &mut reader, &yp("fc1.bias"))?,
                fc2_w: load_t(&content, &mut reader, &yp("fc2.weight"))?,
                fc2_b: load_v(&content, &mut reader, &yp("fc2.bias"))?,
            });
        }

        Ok(Self {
            config,
            encoder_w,
            encoder_b,
            thinking_embed,
            blocks,
            y_encoders,
            head_fc1_w: load_t(&content, &mut reader, "head_fc1.weight")?,
            head_fc1_b: load_v(&content, &mut reader, "head_fc1.bias")?,
            head_fc2_w: load_t(&content, &mut reader, "head_fc2.weight")?,
            head_fc2_b: load_v(&content, &mut reader, "head_fc2.bias")?,
        })
    }

    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (x_s, x_q) = preprocess_x_outer(x_support, x_query, self.config.max_num_features);
        let y_s: Vec<f32> = y_support.iter().map(|&c| c as f32).collect();
        let out = self.forward(&x_s, &y_s, &x_q)?;
        let n_q = x_query.len();
        let width = self.config.max_num_classes + self.config.regression_bin_count;
        let flat: Vec<f32> = to_host(&out)?;
        let mut result = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let row = &flat[i * width..i * width + n_classes];
            let log_probs = log_softmax(row);
            result.push(softmax_host(&log_probs));
        }
        Ok(result)
    }

    pub fn predict_regression(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[f32],
        x_query: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let (x_s, x_q) = preprocess_x_outer(x_support, x_query, self.config.max_num_features);
        let mean_y: f64 = y_support.iter().map(|&v| v as f64).sum::<f64>() / y_support.len() as f64;
        let var_y: f64 = y_support
            .iter()
            .map(|&v| (v as f64 - mean_y).powi(2))
            .sum::<f64>()
            / (y_support.len() - 1) as f64;
        let std_y = var_y.sqrt() + 1e-6;
        let y_s: Vec<f32> = y_support
            .iter()
            .map(|&v| ((v as f64 - mean_y) / std_y) as f32)
            .collect();
        let out = self.forward(&x_s, &y_s, &x_q)?;
        let n_q = x_query.len();
        let width = self.config.max_num_classes + self.config.regression_bin_count;
        let flat: Vec<f32> = to_host(&out)?;
        let n_bins = self.config.regression_bin_count;
        let bin_width =
            (self.config.regression_bin_max - self.config.regression_bin_min) / n_bins as f32;
        let bin_centers: Vec<f32> = (0..n_bins)
            .map(|i| self.config.regression_bin_min + (i as f32 + 0.5) * bin_width)
            .collect();
        let mut result = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let row = &flat[i * width + self.config.max_num_classes..i * width + width];
            let probs = softmax_host(row);
            let expectation: f32 = probs.iter().zip(&bin_centers).map(|(p, c)| p * c).sum();
            result.push((expectation as f64 * std_y + mean_y) as f32);
        }
        Ok(result)
    }

    fn forward(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[f32],
        x_query: &[Vec<f32>],
    ) -> Result<Tensor<2>> {
        let dev = device();
        let n_s = x_support.len();
        let n_q = x_query.len();
        let n_feat = self.config.max_num_features;
        let n_think = self.config.n_thinking_rows;
        let ctx_len = n_think + n_s;

        let mut all_rows: Vec<Vec<f32>> = Vec::with_capacity(n_s + n_q);
        all_rows.extend_from_slice(x_support);
        all_rows.extend_from_slice(x_query);
        let all_rows = clip_outliers(all_rows, n_s, n_feat, CLIP_N_SIGMA);
        let (all_rows, _m, _s) = normalize_data(all_rows, n_s, n_feat);
        let mut all_rows = clip_outliers(all_rows, n_s, n_feat, CLIP_N_SIGMA);
        for row in &mut all_rows {
            for v in row.iter_mut() {
                if v.is_nan() || v.is_infinite() {
                    *v = 0.0;
                }
            }
        }
        let flat: Vec<f32> = all_rows.into_iter().flatten().collect();
        let x_t = Tensor::<2>::from_data(TensorData::new(flat, [n_s + n_q, n_feat]), &dev);
        let x_enc = linear(x_t, self.encoder_w.clone(), self.encoder_b.clone());
        let x_enc = layer_norm_no_affine(x_enc);
        let mut src = Tensor::cat(vec![self.thinking_embed.clone(), x_enc], 0);

        let y_support_t =
            Tensor::<2>::from_data(TensorData::new(y_support.to_vec(), [n_s, 1]), &dev);
        let zeros_think_emb = Tensor::<2>::zeros([n_think, self.config.y_encoder_dim], &dev);
        let kappa = self.config.kappa();
        let beta = attention_beta(kappa, ctx_len, self.config.base_len, self.config.max_len);
        for (blk, yenc) in self.blocks.iter().zip(self.y_encoders.iter()) {
            let y_h = gelu(linear(
                y_support_t.clone(),
                yenc.fc1_w.clone(),
                yenc.fc1_b.clone(),
            ));
            let y_h = linear(y_h, yenc.fc2_w.clone(), yenc.fc2_b.clone());
            let y_support_emb = layer_norm_no_affine(y_h);
            let y_emb = Tensor::cat(vec![zeros_think_emb.clone(), y_support_emb], 0);
            src = layer_forward(src, y_emb, ctx_len, blk, self.config.n_heads, beta as f32);
        }
        let query_out: Tensor<2> = src.narrow(0, ctx_len, n_q);
        let h = gelu(linear(
            query_out,
            self.head_fc1_w.clone(),
            self.head_fc1_b.clone(),
        ));
        Ok(linear_o(
            h,
            self.head_fc2_w.clone(),
            Some(self.head_fc2_b.clone()),
        ))
    }
}

fn layer_forward(
    src: Tensor<2>,
    y_emb: Tensor<2>,
    ctx_len: usize,
    blk: &BurnBlockW,
    n_heads: usize,
    beta: f32,
) -> Tensor<2> {
    let dim = src.dims()[1];
    let head_dim = dim / n_heads;
    let l = src.dims()[0];
    let h = layer_norm(
        src.clone(),
        blk.attn_norm_w.clone(),
        blk.attn_norm_b.clone(),
    );
    let q = linear_nd(h.clone(), blk.q_proj_w.clone(), None);
    let gate = sigmoid(linear_nd(q.clone(), blk.q_gate_w.clone(), None));
    let h_ctx: Tensor<2> = h.narrow(0, 0, ctx_len);
    let k = linear_nd(h_ctx.clone(), blk.k_proj_w.clone(), None);
    let v_in = Tensor::cat(vec![h_ctx, y_emb], 1);
    let v = linear_nd(v_in, blk.v_proj_w.clone(), None);
    let q: Tensor<3> = q.reshape([l, n_heads, head_dim]).permute([1, 0, 2]);
    let k: Tensor<3> = k.reshape([ctx_len, n_heads, head_dim]).permute([1, 0, 2]);
    let v: Tensor<3> = v.reshape([ctx_len, n_heads, head_dim]).permute([1, 0, 2]);
    let q = rms(q, blk.q_norm_w.clone());
    let k = rms(k, blk.k_norm_w.clone());
    let scale = beta / (head_dim as f32).sqrt();
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let attn = softmax_last(scores);
    let out = attn.matmul(v);
    let out: Tensor<3> = out.permute([1, 0, 2]);
    let gate: Tensor<3> = gate.reshape([l, n_heads, 1]);
    let out = out * gate;
    let out: Tensor<2> = out.reshape([l, dim]);
    let attn_out = linear_nd(out, blk.out_proj_w.clone(), None);
    let x1 = src + attn_out;
    let ff_in = layer_norm(x1.clone(), blk.ff_norm_w.clone(), blk.ff_norm_b.clone());
    let up = linear_nd(ff_in, blk.ff_up_w.clone(), None);
    let ff_dim2 = up.dims()[1] / 2;
    let u: Tensor<2> = up.clone().narrow(1, 0, ff_dim2);
    let v_ff: Tensor<2> = up.narrow(1, ff_dim2, ff_dim2);
    let ff_out = linear_nd(silu(u) * v_ff, blk.ff_down_w.clone(), None);
    x1 + ff_out
}

fn attention_beta(kappa: Option<f64>, ctx_len: usize, base_len: usize, max_len: usize) -> f64 {
    match kappa {
        None => 1.0,
        Some(kappa) => {
            let n = (ctx_len as f64).min(max_len as f64);
            let log_term = (n / base_len as f64).ln().max(0.0);
            1.0 + kappa * log_term
        }
    }
}

fn softmax_host(logits: &[f32]) -> Vec<f32> {
    zsfm_core::softmax(logits)
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&v| v - max - log_sum_exp).collect()
}

fn preprocess_x_outer(
    x_support: &[Vec<f32>],
    x_query: &[Vec<f32>],
    max_features: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_feat = x_support[0].len();
    let mut impute_mean = vec![0f32; n_feat];
    for f in 0..n_feat {
        let mut sum = 0f64;
        let mut count = 0usize;
        for row in x_support {
            if !row[f].is_nan() {
                sum += row[f] as f64;
                count += 1;
            }
        }
        impute_mean[f] = if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        };
    }
    let impute = |row: &[f32]| -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(f, &v)| if v.is_nan() { impute_mean[f] } else { v })
            .collect()
    };
    let support_imputed: Vec<Vec<f32>> = x_support.iter().map(|r| impute(r)).collect();
    let query_imputed: Vec<Vec<f32>> = x_query.iter().map(|r| impute(r)).collect();
    let n_s = support_imputed.len();
    let mut scaler_mean = vec![0f32; n_feat];
    let mut scaler_std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let mean: f64 = support_imputed.iter().map(|r| r[f] as f64).sum::<f64>() / n_s as f64;
        let var: f64 = support_imputed
            .iter()
            .map(|r| (r[f] as f64 - mean).powi(2))
            .sum::<f64>()
            / n_s as f64;
        scaler_mean[f] = mean as f32;
        scaler_std[f] = if var == 0.0 { 1.0 } else { var.sqrt() as f32 };
    }
    let scale_and_pad = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.iter()
            .map(|row| {
                let mut out = vec![0f32; max_features];
                for f in 0..n_feat.min(max_features) {
                    out[f] = (row[f] - scaler_mean[f]) / scaler_std[f];
                }
                out
            })
            .collect()
    };
    (
        scale_and_pad(&support_imputed),
        scale_and_pad(&query_imputed),
    )
}

fn clip_outliers(
    mut rows: Vec<Vec<f32>>,
    n_ctx: usize,
    n_feat: usize,
    n_sigma: f32,
) -> Vec<Vec<f32>> {
    for f in 0..n_feat {
        let ctx_vals: Vec<f32> = rows[..n_ctx]
            .iter()
            .map(|r| r[f])
            .filter(|v| !v.is_nan())
            .collect();
        if ctx_vals.is_empty() {
            continue;
        }
        let mean = ctx_vals.iter().sum::<f32>() / ctx_vals.len() as f32;
        let std1 = population_std(&ctx_vals, mean);
        let cutoff1 = n_sigma * std1;
        let refined: Vec<f32> = ctx_vals
            .iter()
            .copied()
            .filter(|&v| (v - mean).abs() <= cutoff1)
            .collect();
        let std2 = if refined.len() < 2 {
            std1
        } else {
            let refined_mean = refined.iter().sum::<f32>() / refined.len() as f32;
            population_std(&refined, refined_mean)
        };
        let cutoff2 = n_sigma * std2;
        let (lo, hi) = (mean - cutoff2, mean + cutoff2);
        for row in rows.iter_mut() {
            row[f] = row[f].clamp(lo, hi);
        }
    }
    rows
}

fn normalize_data(
    rows: Vec<Vec<f32>>,
    n_ctx: usize,
    n_feat: usize,
) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let ctx_vals: Vec<f32> = rows[..n_ctx]
            .iter()
            .map(|r| r[f])
            .filter(|v| !v.is_nan())
            .collect();
        if ctx_vals.is_empty() {
            mean[f] = 0.0;
            std[f] = 1e-6;
            continue;
        }
        let m = ctx_vals.iter().sum::<f32>() / ctx_vals.len() as f32;
        mean[f] = m;
        std[f] = population_std(&ctx_vals, m) + 1e-6;
    }
    let out: Vec<Vec<f32>> = rows
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(f, &v)| (v - mean[f]) / std[f])
                .collect()
        })
        .collect();
    (out, mean, std)
}

fn population_std(vals: &[f32], mean: f32) -> f32 {
    if vals.len() < 2 {
        return 0.0;
    }
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (vals.len() - 1) as f32;
    var.sqrt()
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

    fn tiny_config() -> TabDptConfig {
        TabDptConfig {
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            ff_dim: 8,
            y_encoder_dim: 4,
            max_num_classes: 2,
            regression_bin_count: 4,
            regression_bin_min: -2.0,
            regression_bin_max: 2.0,
            max_num_features: 4,
            base_len: 64,
            max_len: 64,
            n_thinking_rows: 2,
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
    fn burn_matches_candle_classification() {
        let dir = std::env::temp_dir().join(format!("tabdpt-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tabdpt".into()),
        );
        let mut seed = 1900u64;
        let s = 0.2;
        put2(&mut w, "encoder.weight", 8, 4, &mut seed, s);
        put1(&mut w, "encoder.bias", 8, &mut seed);
        put2(&mut w, "thinking_embed", 2, 8, &mut seed, s);
        put1(&mut w, "blk.0.attn_norm.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_norm.bias", 8, &mut seed);
        put1(&mut w, "blk.0.ff_norm.weight", 8, &mut seed);
        put1(&mut w, "blk.0.ff_norm.bias", 8, &mut seed);
        put2(&mut w, "blk.0.q_proj.weight", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.k_proj.weight", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.v_proj.weight", 8, 12, &mut seed, s);
        put2(&mut w, "blk.0.out_proj.weight", 8, 8, &mut seed, s);
        put2(&mut w, "blk.0.q_gate.weight", 2, 8, &mut seed, s);
        put1(&mut w, "blk.0.q_norm.weight", 4, &mut seed);
        put1(&mut w, "blk.0.k_norm.weight", 4, &mut seed);
        put2(&mut w, "blk.0.ff_up.weight", 16, 8, &mut seed, s);
        put2(&mut w, "blk.0.ff_down.weight", 8, 8, &mut seed, s);
        put2(&mut w, "y_enc.0.fc1.weight", 4, 1, &mut seed, s);
        put1(&mut w, "y_enc.0.fc1.bias", 4, &mut seed);
        put2(&mut w, "y_enc.0.fc2.weight", 4, 4, &mut seed, s);
        put1(&mut w, "y_enc.0.fc2.bias", 4, &mut seed);
        put2(&mut w, "head_fc1.weight", 8, 8, &mut seed, s);
        put1(&mut w, "head_fc1.bias", 8, &mut seed);
        put2(&mut w, "head_fc2.weight", 6, 8, &mut seed, s);
        put1(&mut w, "head_fc2.bias", 6, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::TabDptModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnTabDptModel::load(&path, cfg).unwrap();
        let x_support = vec![
            vec![0.1, 1.2],
            vec![0.9, -0.3],
            vec![-1.1, 0.4],
            vec![1.5, 1.1],
        ];
        let y_support = vec![0usize, 1, 1, 0];
        let x_query = vec![vec![0.2, 0.9], vec![-0.8, 0.1]];
        let t0 = std::time::Instant::now();
        let a = candle
            .predict_classification(&x_support, &y_support, &x_query, 2)
            .unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn
            .predict_classification(&x_support, &y_support, &x_query, 2)
            .unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut err = 0.0f32;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                err = err.max((x - y).abs());
            }
        }
        println!("tabdpt synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "tabdpt Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
