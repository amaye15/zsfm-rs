//! TabDPT inference engine — zero-shot forward pass only, single-pass (no ensembling; upstream
//! defaults to averaging 8 class-permuted passes, but `n_ensembles=1` is a genuine, documented
//! mode in the reference implementation, not a shortcut — see `classifier.py`'s
//! `predict()`/`ensemble_predict_proba()` split).
//!
//! Architecture: 32-layer transformer over `[thinking rows][support rows][query rows]`. Each
//! layer's attention lets every position attend to the context (thinking + support) only, with
//! a per-layer y-embedding (a small MLP, re-run per layer) folded into V, RMSNorm'd Q/K, a
//! length-adaptive temperature scale, and a sigmoid output gate per head. One checkpoint serves
//! both tasks — the head produces `max_num_classes + regression_bin_count` outputs; classifier
//! reads the first `n_classes`, regressor reads the rest as a binned distribution over
//! `[regression_bin_min, regression_bin_max]`.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use crate::config::TabDptConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct BlockW {
    attn_norm_w: Tensor,
    attn_norm_b: Tensor,
    ff_norm_w: Tensor,
    ff_norm_b: Tensor,
    q_proj_w: Tensor,
    k_proj_w: Tensor,
    v_proj_w: Tensor,
    out_proj_w: Tensor,
    q_gate_w: Tensor,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    ff_up_w: Tensor,
    ff_down_w: Tensor,
}

struct YEncW {
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

pub struct TabDptModel {
    device: Device,
    config: TabDptConfig,
    encoder_w: Tensor,
    encoder_b: Tensor,
    thinking_embed: Tensor,
    blocks: Vec<BlockW>,
    y_encoders: Vec<YEncW>,
    head_fc1_w: Tensor,
    head_fc1_b: Tensor,
    head_fc2_w: Tensor,
    head_fc2_b: Tensor,
}

const LN_EPS: f64 = 1e-5;
const RMS_EPS: f64 = 1e-8;
const CLIP_N_SIGMA: f32 = 8.0;

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

fn load_t(content: &gguf_file::Content, reader: &mut (impl Read + Seek), name: &str, device: &Device) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

impl TabDptModel {
    pub fn load(gguf_path: &Path, config: TabDptConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let encoder_w = load_t(&content, &mut reader, "encoder.weight", &device)?;
        let encoder_b = load_t(&content, &mut reader, "encoder.bias", &device)?;
        let thinking_embed = load_t(&content, &mut reader, "thinking_embed", &device)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        let mut y_encoders = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            blocks.push(BlockW {
                attn_norm_w: load_t(&content, &mut reader, &p("attn_norm.weight"), &device)?,
                attn_norm_b: load_t(&content, &mut reader, &p("attn_norm.bias"), &device)?,
                ff_norm_w: load_t(&content, &mut reader, &p("ff_norm.weight"), &device)?,
                ff_norm_b: load_t(&content, &mut reader, &p("ff_norm.bias"), &device)?,
                q_proj_w: load_t(&content, &mut reader, &p("q_proj.weight"), &device)?,
                k_proj_w: load_t(&content, &mut reader, &p("k_proj.weight"), &device)?,
                v_proj_w: load_t(&content, &mut reader, &p("v_proj.weight"), &device)?,
                out_proj_w: load_t(&content, &mut reader, &p("out_proj.weight"), &device)?,
                q_gate_w: load_t(&content, &mut reader, &p("q_gate.weight"), &device)?,
                q_norm_w: load_t(&content, &mut reader, &p("q_norm.weight"), &device)?,
                k_norm_w: load_t(&content, &mut reader, &p("k_norm.weight"), &device)?,
                ff_up_w: load_t(&content, &mut reader, &p("ff_up.weight"), &device)?,
                ff_down_w: load_t(&content, &mut reader, &p("ff_down.weight"), &device)?,
            });

            let yp = |s: &str| format!("y_enc.{n}.{s}");
            y_encoders.push(YEncW {
                fc1_w: load_t(&content, &mut reader, &yp("fc1.weight"), &device)?,
                fc1_b: load_t(&content, &mut reader, &yp("fc1.bias"), &device)?,
                fc2_w: load_t(&content, &mut reader, &yp("fc2.weight"), &device)?,
                fc2_b: load_t(&content, &mut reader, &yp("fc2.bias"), &device)?,
            });
        }

        let head_fc1_w = load_t(&content, &mut reader, "head_fc1.weight", &device)?;
        let head_fc1_b = load_t(&content, &mut reader, "head_fc1.bias", &device)?;
        let head_fc2_w = load_t(&content, &mut reader, "head_fc2.weight", &device)?;
        let head_fc2_b = load_t(&content, &mut reader, "head_fc2.bias", &device)?;

        Ok(Self {
            device,
            config,
            encoder_w,
            encoder_b,
            thinking_embed,
            blocks,
            y_encoders,
            head_fc1_w,
            head_fc1_b,
            head_fc2_w,
            head_fc2_b,
        })
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Zero-shot classification. `y_support` are class indices `0..n_classes`. Returns
    /// probabilities `[n_query][n_classes]` — note the reference implementation applies
    /// `softmax(log_softmax(logits))` (not a plain single softmax); replicated exactly.
    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let (x_s, x_q) = preprocess_x_outer(x_support, x_query, self.config.max_num_features);
        let y_s: Vec<f32> = y_support.iter().map(|&c| c as f32).collect();

        let out = self.forward(&x_s, &y_s, &x_q)?; // (n_q, max_num_classes + regression_bin_count)
        let n_q = x_query.len();
        let width = self.config.max_num_classes + self.config.regression_bin_count;
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;

        let mut result = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let row = &flat[i * width..i * width + n_classes];
            let log_probs = log_softmax(row);
            result.push(softmax(&log_probs));
        }
        Ok(result)
    }

    /// Zero-shot regression. Returns predicted values in `y_support`'s original scale.
    pub fn predict_regression(&self, x_support: &[Vec<f32>], y_support: &[f32], x_query: &[Vec<f32>]) -> Result<Vec<f32>> {
        let (x_s, x_q) = preprocess_x_outer(x_support, x_query, self.config.max_num_features);

        let mean_y: f64 = y_support.iter().map(|&v| v as f64).sum::<f64>() / y_support.len() as f64;
        let var_y: f64 = y_support.iter().map(|&v| (v as f64 - mean_y).powi(2)).sum::<f64>() / (y_support.len() - 1) as f64;
        let std_y = var_y.sqrt() + 1e-6;
        let y_s: Vec<f32> = y_support.iter().map(|&v| ((v as f64 - mean_y) / std_y) as f32).collect();

        let out = self.forward(&x_s, &y_s, &x_q)?;
        let n_q = x_query.len();
        let width = self.config.max_num_classes + self.config.regression_bin_count;
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;

        let n_bins = self.config.regression_bin_count;
        let bin_lo = self.config.regression_bin_min;
        let bin_hi = self.config.regression_bin_max;
        let bin_width = (bin_hi - bin_lo) / n_bins as f32;
        let bin_centers: Vec<f32> = (0..n_bins).map(|i| bin_lo + (i as f32 + 0.5) * bin_width).collect();

        let mut result = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let row = &flat[i * width + self.config.max_num_classes..i * width + width];
            let probs = softmax(row);
            let expectation: f32 = probs.iter().zip(&bin_centers).map(|(p, c)| p * c).sum();
            result.push((expectation as f64 * std_y + mean_y) as f32);
        }
        Ok(result)
    }

    /// `x_support`/`x_query` are already outer-preprocessed (imputed, standard-scaled, padded to
    /// `max_num_features`). `y_support` is already the scalar fed to the model (class index for
    /// classification, z-scored target for regression). Returns `[n_query][max_num_classes +
    /// regression_bin_count]` raw head outputs.
    fn forward(&self, x_support: &[Vec<f32>], y_support: &[f32], x_query: &[Vec<f32>]) -> Result<Tensor> {
        let n_s = x_support.len();
        let n_q = x_query.len();
        let n_feat = self.config.max_num_features;
        let n_think = self.config.n_thinking_rows;
        let ctx_len = n_think + n_s;

        // Combine support+query rows, model-internal clip/normalize (context-fit, applied to all).
        let mut all_rows: Vec<Vec<f32>> = Vec::with_capacity(n_s + n_q);
        all_rows.extend_from_slice(x_support);
        all_rows.extend_from_slice(x_query);
        let all_rows = clip_outliers(all_rows, n_s, n_feat, CLIP_N_SIGMA);
        let (all_rows, _mean, _std) = normalize_data(all_rows, n_s, n_feat);
        let mut all_rows = clip_outliers(all_rows, n_s, n_feat, CLIP_N_SIGMA);
        for row in &mut all_rows {
            for v in row.iter_mut() {
                if v.is_nan() || v.is_infinite() {
                    *v = 0.0;
                }
            }
        }

        let flat: Vec<f32> = all_rows.into_iter().flatten().collect();
        let x_t = Tensor::from_vec(flat, (n_s + n_q, n_feat), &self.device)?;
        let x_enc = zsfm_nn::linear_bias(&x_t, &self.encoder_w, &self.encoder_b)?; // (T, dim)
        let x_enc = layer_norm_no_affine(&x_enc, LN_EPS)?;

        let mut src = Tensor::cat(&[&self.thinking_embed, &x_enc], 0)?; // (n_think+T, dim)

        let y_support_t = Tensor::from_vec(y_support.to_vec(), (n_s, 1), &self.device)?;
        let zeros_think_emb = Tensor::zeros((n_think, self.config.y_encoder_dim), DType::F32, &self.device)?;

        let kappa = self.config.kappa();
        let beta = attention_beta(kappa, ctx_len, self.config.base_len, self.config.max_len);

        for (blk, yenc) in self.blocks.iter().zip(self.y_encoders.iter()) {
            // Encode the *actual* support targets first, then prepend a true zero embedding for
            // the thinking rows — NOT the other way around: the MLP has biases, so `MLP(0)` is
            // not the zero vector, and the reference concatenates zeros only *after* encoding.
            let y_h = zsfm_nn::linear_bias(&y_support_t, &yenc.fc1_w, &yenc.fc1_b)?.gelu_erf()?;
            let y_h = zsfm_nn::linear_bias(&y_h, &yenc.fc2_w, &yenc.fc2_b)?;
            let y_support_emb = layer_norm_no_affine(&y_h, LN_EPS)?; // (n_s, y_encoder_dim)
            let y_emb = Tensor::cat(&[&zeros_think_emb, &y_support_emb], 0)?; // (ctx_len, y_encoder_dim)

            src = layer_forward(&src, &y_emb, ctx_len, blk, self.config.n_heads, beta)?;
        }

        let query_out = src.narrow(0, ctx_len, n_q)?; // (n_q, dim)
        let h = zsfm_nn::linear_bias(&query_out, &self.head_fc1_w, &self.head_fc1_b)?.gelu_erf()?;
        zsfm_nn::linear_bias(&h, &self.head_fc2_w, &self.head_fc2_b).map_err(anyhow::Error::from)
    }
}

/// One `TransformerEncoderLayer`: attn_norm -> gated RMSNorm'd attention (context-only K/V,
/// all-position Q) -> residual -> ff_norm -> SwiGLU -> residual.
fn layer_forward(src: &Tensor, y_emb: &Tensor, ctx_len: usize, blk: &BlockW, n_heads: usize, beta: f64) -> Result<Tensor> {
    let dim = src.dim(1)?;
    let head_dim = dim / n_heads;
    let l = src.dim(0)?;

    let h = zsfm_nn::layer_norm(src, &blk.attn_norm_w, &blk.attn_norm_b, LN_EPS)?;
    let q = zsfm_nn::linear_nobias(&h, &blk.q_proj_w)?; // (L, dim)
    let gate = candle_nn::ops::sigmoid(&zsfm_nn::linear_nobias(&q, &blk.q_gate_w)?)?; // (L, n_heads)

    let h_ctx = h.narrow(0, 0, ctx_len)?;
    let k = zsfm_nn::linear_nobias(&h_ctx, &blk.k_proj_w)?; // (ctx_len, dim)
    let v_in = Tensor::cat(&[&h_ctx, y_emb], 1)?; // (ctx_len, dim + y_encoder_dim)
    let v = zsfm_nn::linear_nobias(&v_in, &blk.v_proj_w)?; // (ctx_len, dim)

    let q = q.reshape((l, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?; // (h, L, hd)
    let k = k.reshape((ctx_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;
    let v = v.reshape((ctx_len, n_heads, head_dim))?.permute((1, 0, 2))?.contiguous()?;

    let q = zsfm_nn::rms_norm(&q, Some(&blk.q_norm_w), RMS_EPS)?;
    let k = zsfm_nn::rms_norm(&k, Some(&blk.k_norm_w), RMS_EPS)?;

    let scale = beta / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?; // (h, L, ctx_len)
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = attn.matmul(&v)?; // (h, L, hd)
    let out = out.permute((1, 0, 2))?.contiguous()?; // (L, h, hd)

    let gate = gate.reshape((l, n_heads, 1))?;
    let out = out.broadcast_mul(&gate)?.reshape((l, dim))?;
    let attn_out = zsfm_nn::linear_nobias(&out, &blk.out_proj_w)?;

    let x1 = (src + attn_out)?;
    let ff_in = zsfm_nn::layer_norm(&x1, &blk.ff_norm_w, &blk.ff_norm_b, LN_EPS)?;
    let up = zsfm_nn::linear_nobias(&ff_in, &blk.ff_up_w)?;
    let ff_dim2 = up.dim(1)? / 2;
    let u = up.narrow(1, 0, ff_dim2)?;
    let v_ff = up.narrow(1, ff_dim2, ff_dim2)?;
    let ff_out = zsfm_nn::linear_nobias(&(u.silu()? * v_ff)?, &blk.ff_down_w)?;

    Ok((x1 + ff_out)?)
}

/// `1 + kappa * max(0, ln(min(ctx_len, max_len) / base_len))`, or `1.0` if attention scaling is
/// disabled (`base_len == max_len`).
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

fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    Ok(centered.broadcast_div(&std)?)
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&v| (v - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&v| v - max - log_sum_exp).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

// ---------------------------------------------------------------------------
// Outer preprocessing (SimpleImputer(mean) + StandardScaler, fit on support, applied to both,
// then zero-padded to `max_num_features`)
// ---------------------------------------------------------------------------

fn preprocess_x_outer(x_support: &[Vec<f32>], x_query: &[Vec<f32>], max_features: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
        impute_mean[f] = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
    }
    let impute = |row: &[f32]| -> Vec<f32> {
        row.iter().enumerate().map(|(f, &v)| if v.is_nan() { impute_mean[f] } else { v }).collect()
    };
    let support_imputed: Vec<Vec<f32>> = x_support.iter().map(|r| impute(r)).collect();
    let query_imputed: Vec<Vec<f32>> = x_query.iter().map(|r| impute(r)).collect();

    // StandardScaler stats, fit on the (now NaN-free) support data.
    let n_s = support_imputed.len();
    let mut scaler_mean = vec![0f32; n_feat];
    let mut scaler_std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let mean: f64 = support_imputed.iter().map(|r| r[f] as f64).sum::<f64>() / n_s as f64;
        let var: f64 = support_imputed.iter().map(|r| (r[f] as f64 - mean).powi(2)).sum::<f64>() / n_s as f64;
        scaler_mean[f] = mean as f32;
        // sklearn StandardScaler: zero-variance columns get scale=1 (no-op) instead of divide-by-zero.
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

    (scale_and_pad(&support_imputed), scale_and_pad(&query_imputed))
}

// ---------------------------------------------------------------------------
// Model-internal preprocessing (utils.py's `clip_outliers`/`normalize_data`), fit on the
// context (first `n_ctx` rows), applied to all rows, per feature column.
// ---------------------------------------------------------------------------

fn clip_outliers(mut rows: Vec<Vec<f32>>, n_ctx: usize, n_feat: usize, n_sigma: f32) -> Vec<Vec<f32>> {
    for f in 0..n_feat {
        let ctx_vals: Vec<f32> = rows[..n_ctx].iter().map(|r| r[f]).filter(|v| !v.is_nan()).collect();
        if ctx_vals.is_empty() {
            continue;
        }
        // `maskstd` recomputes its own mean from whatever mask it's given each call — the second
        // pass's std is centered on the *refined* subset's mean, not the original mean. The
        // final clip bounds, though, stay centered on the *original* mean (matches `utils.py`
        // exactly: `torch.clip(data, mean - cutoff, mean + cutoff)` uses the first `mean`).
        let mean = ctx_vals.iter().sum::<f32>() / ctx_vals.len() as f32;
        let std1 = population_std(&ctx_vals, mean);
        let cutoff1 = n_sigma * std1;
        let refined: Vec<f32> = ctx_vals.iter().copied().filter(|&v| (v - mean).abs() <= cutoff1).collect();
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

fn normalize_data(rows: Vec<Vec<f32>>, n_ctx: usize, n_feat: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let ctx_vals: Vec<f32> = rows[..n_ctx].iter().map(|r| r[f]).filter(|v| !v.is_nan()).collect();
        if ctx_vals.is_empty() {
            mean[f] = 0.0;
            std[f] = 1e-6;
            continue;
        }
        let m = ctx_vals.iter().sum::<f32>() / ctx_vals.len() as f32;
        mean[f] = m;
        std[f] = population_std(&ctx_vals, m) + 1e-6;
    }
    let out: Vec<Vec<f32>> =
        rows.into_iter().map(|row| row.iter().enumerate().map(|(f, &v)| (v - mean[f]) / std[f]).collect()).collect();
    (out, mean, std)
}

fn population_std(vals: &[f32], mean: f32) -> f32 {
    if vals.len() < 2 {
        return 0.0;
    }
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (vals.len() - 1) as f32;
    var.sqrt()
}
