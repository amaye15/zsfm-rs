//! Mitra (Tab2D) inference engine — zero-shot forward pass only (no fine-tuning; see the crate
//! README for why). Architecture: per-feature quantile-bucketize embedding → prepend a learned
//! y-embedding as an extra "feature" column → 12 layers of (row self/cross-attention → MLP →
//! feature self-attention → MLP) → final LayerNorm + linear head, read out at the y-column.
//!
//! Two deliberate, documented divergences from AutoGluon's default `MitraClassifier`/
//! `MitraRegressor` (both scope decisions, not omissions):
//! - `random_mirror_x`/`random_mirror_regression` (default ON upstream) are OFF here for
//!   determinism — even upstream they draw from the *unseeded* global NumPy RNG, so upstream's
//!   own "default" behavior isn't reproducible either without external global-seed control.
//! - The per-call support-row shuffle (`np.random.RandomState.choice`, seeded) is not
//!   replicated — same scope call already made for TabFM's OOF K-fold splitting (see
//!   `tabfm/src/ensemble/oof.rs`): porting NumPy's legacy `RandomState` exactly is extra work
//!   for a step the model is mathematically invariant to (attention has no row positional
//!   encoding); rows are fed in the given order.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::config::{MitraConfig, Task};

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct AttnW {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,
}

struct BlockW {
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ln3_w: Tensor,
    ln3_b: Tensor,
    ln4_w: Tensor,
    ln4_b: Tensor,
    attn_row: AttnW,
    attn_feat: AttnW,
    mlp1_fc1_w: Tensor,
    mlp1_fc1_b: Tensor,
    mlp1_fc2_w: Tensor,
    mlp1_fc2_b: Tensor,
    mlp2_fc1_w: Tensor,
    mlp2_fc1_b: Tensor,
    mlp2_fc2_w: Tensor,
    mlp2_fc2_b: Tensor,
}

pub struct MitraModel {
    device: Device,
    config: MitraConfig,
    x_embed_w: Tensor, // [dim, 1]
    x_embed_b: Tensor, // [dim]
    y_embed_w: Tensor, // classifier: [dim_output, dim] (Embedding table); regressor: [dim, 1] (Linear weight)
    y_embed_b: Option<Tensor>, // regressor only: [dim]
    y_mask_w: Tensor,  // [1, dim]
    blocks: Vec<BlockW>,
    norm_f_w: Tensor,
    norm_f_b: Tensor,
    head_w: Tensor,
    head_b: Tensor,
}

const LN_EPS: f64 = 1e-5;

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

fn load_t(content: &gguf_file::Content, reader: &mut (impl Read + Seek), name: &str, device: &Device) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<AttnW> {
    let mut t = |s: &str| load_t(content, reader, &format!("{prefix}.{s}"), device);
    Ok(AttnW {
        q_w: t("q.weight")?,
        q_b: t("q.bias")?,
        k_w: t("k.weight")?,
        k_b: t("k.bias")?,
        v_w: t("v.weight")?,
        v_b: t("v.bias")?,
        o_w: t("o.weight")?,
        o_b: t("o.bias")?,
    })
}

impl MitraModel {
    pub fn load(gguf_path: &Path, config: MitraConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let x_embed_w = load_t(&content, &mut reader, "x_embed.weight", &device)?;
        let x_embed_b = load_t(&content, &mut reader, "x_embed.bias", &device)?;
        let y_embed_w = load_t(&content, &mut reader, "y_embed.weight", &device)?;
        let y_embed_b = match config.task {
            Task::Regression => Some(load_t(&content, &mut reader, "y_embed.bias", &device)?),
            Task::Classification => None,
        };
        let y_mask_w = load_t(&content, &mut reader, "y_mask.weight", &device)?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let attn_row = load_attn(&content, &mut reader, &p("attn_row"), &device)?;
            let attn_feat = load_attn(&content, &mut reader, &p("attn_feat"), &device)?;
            blocks.push(BlockW {
                ln1_w: load_t(&content, &mut reader, &p("ln1.weight"), &device)?,
                ln1_b: load_t(&content, &mut reader, &p("ln1.bias"), &device)?,
                ln2_w: load_t(&content, &mut reader, &p("ln2.weight"), &device)?,
                ln2_b: load_t(&content, &mut reader, &p("ln2.bias"), &device)?,
                ln3_w: load_t(&content, &mut reader, &p("ln3.weight"), &device)?,
                ln3_b: load_t(&content, &mut reader, &p("ln3.bias"), &device)?,
                ln4_w: load_t(&content, &mut reader, &p("ln4.weight"), &device)?,
                ln4_b: load_t(&content, &mut reader, &p("ln4.bias"), &device)?,
                attn_row,
                attn_feat,
                mlp1_fc1_w: load_t(&content, &mut reader, &p("mlp1_fc1.weight"), &device)?,
                mlp1_fc1_b: load_t(&content, &mut reader, &p("mlp1_fc1.bias"), &device)?,
                mlp1_fc2_w: load_t(&content, &mut reader, &p("mlp1_fc2.weight"), &device)?,
                mlp1_fc2_b: load_t(&content, &mut reader, &p("mlp1_fc2.bias"), &device)?,
                mlp2_fc1_w: load_t(&content, &mut reader, &p("mlp2_fc1.weight"), &device)?,
                mlp2_fc1_b: load_t(&content, &mut reader, &p("mlp2_fc1.bias"), &device)?,
                mlp2_fc2_w: load_t(&content, &mut reader, &p("mlp2_fc2.weight"), &device)?,
                mlp2_fc2_b: load_t(&content, &mut reader, &p("mlp2_fc2.bias"), &device)?,
            });
        }

        let norm_f_w = load_t(&content, &mut reader, "norm_f.weight", &device)?;
        let norm_f_b = load_t(&content, &mut reader, "norm_f.bias", &device)?;
        let head_w = load_t(&content, &mut reader, "head.weight", &device)?;
        let head_b = load_t(&content, &mut reader, "head.bias", &device)?;

        Ok(Self {
            device,
            config,
            x_embed_w,
            x_embed_b,
            y_embed_w,
            y_embed_b,
            y_mask_w,
            blocks,
            norm_f_w,
            norm_f_b,
            head_w,
            head_b,
        })
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Zero-shot classification. `y_support` are class indices `0..n_classes`. Returns raw
    /// logits `[n_query][n_classes]` (softmax if you want probabilities).
    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(self.config.task == Task::Classification, "model was loaded as a regressor");
        let (x_s, x_q, kept) = preprocess_x(x_support, x_query);
        let n_s = x_s.len();
        let n_q = x_q.len();
        let n_feat = kept.len();

        let x_emb = self.embed_x(&x_s, &x_q, n_s, n_q, n_feat)?;
        let y_support_emb = self.embed_y_classes_support(y_support, n_s)?;
        let y_query_mask = self.embed_y_mask(n_q)?;

        let logits = self.forward(x_emb, y_support_emb, y_query_mask, n_s, n_q, n_feat)?;
        let logits: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let dim_out = self.config.dim_output;
        Ok((0..n_q).map(|i| logits[i * dim_out..i * dim_out + n_classes].to_vec()).collect())
    }

    /// Zero-shot regression. Returns predicted values in `y_support`'s original scale.
    pub fn predict_regression(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[f32],
        x_query: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(self.config.task == Task::Regression, "model was loaded as a classifier");
        let (x_s, x_q, kept) = preprocess_x(x_support, x_query);
        let n_s = x_s.len();
        let n_q = x_q.len();
        let n_feat = kept.len();

        let y_min = y_support.iter().copied().fold(f32::INFINITY, f32::min);
        let y_max = y_support.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        anyhow::ensure!(y_max > y_min, "y_support must have at least two distinct values");
        let y_scaled: Vec<f32> = y_support.iter().map(|&v| (v - y_min) / (y_max - y_min)).collect();

        let x_emb = self.embed_x(&x_s, &x_q, n_s, n_q, n_feat)?;
        let y_support_emb = self.embed_y_regression_support(&y_scaled)?;
        let y_query_mask = self.embed_y_mask(n_q)?;

        let out = self.forward(x_emb, y_support_emb, y_query_mask, n_s, n_q, n_feat)?;
        let out: Vec<f32> = out.flatten_all()?.to_vec1()?;
        Ok(out.into_iter().map(|v| v * (y_max - y_min) + y_min).collect())
    }

    fn embed_x(&self, x_s: &[Vec<f32>], x_q: &[Vec<f32>], n_s: usize, n_q: usize, n_feat: usize) -> Result<(Tensor, Tensor)> {
        let (bx_s, bx_q) = quantile_bucketize_normalize(x_s, x_q);
        let flat_s: Vec<f32> = bx_s.into_iter().flatten().collect();
        let flat_q: Vec<f32> = bx_q.into_iter().flatten().collect();

        let x_s_t = Tensor::from_vec(flat_s, (n_s * n_feat, 1), &self.device)?;
        let x_q_t = Tensor::from_vec(flat_q, (n_q * n_feat, 1), &self.device)?;

        let x_s_emb = zsfm_nn::linear_bias(&x_s_t, &self.x_embed_w, &self.x_embed_b)?.reshape((n_s, n_feat, self.config.dim))?;
        let x_q_emb = zsfm_nn::linear_bias(&x_q_t, &self.x_embed_w, &self.x_embed_b)?.reshape((n_q, n_feat, self.config.dim))?;
        Ok((x_s_emb, x_q_emb))
    }

    fn embed_y_classes_support(&self, y_support: &[usize], n_s: usize) -> Result<Tensor> {
        let dim = self.config.dim;
        let table: Vec<f32> = self.y_embed_w.flatten_all()?.to_vec1()?;
        let mut out = vec![0f32; n_s * dim];
        for (i, &cls) in y_support.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&table[cls * dim..(cls + 1) * dim]);
        }
        Ok(Tensor::from_vec(out, (n_s, 1, dim), &self.device)?)
    }

    fn embed_y_regression_support(&self, y_scaled: &[f32]) -> Result<Tensor> {
        let n_s = y_scaled.len();
        let y_t = Tensor::from_vec(y_scaled.to_vec(), (n_s, 1), &self.device)?;
        let b = self.y_embed_b.as_ref().context("regressor y_embed missing bias")?;
        let emb = zsfm_nn::linear_bias(&y_t, &self.y_embed_w, b)?; // (n_s, dim)
        Ok(emb.reshape((n_s, 1, self.config.dim))?)
    }

    fn embed_y_mask(&self, n_q: usize) -> Result<Tensor> {
        let dim = self.config.dim;
        let row: Vec<f32> = self.y_mask_w.flatten_all()?.to_vec1()?; // [dim]
        let mut out = Vec::with_capacity(n_q * dim);
        for _ in 0..n_q {
            out.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(out, (n_q, 1, dim), &self.device)?)
    }

    /// `x_emb` = (support_x_emb, query_x_emb), each `[n, n_feat, dim]`.
    /// `y_support_emb`: `[n_s, 1, dim]`. `y_query_mask`: `[n_q, 1, dim]`.
    /// Returns `[n_q, dim_output]` (the model's output read out at the y-column).
    fn forward(
        &self,
        x_emb: (Tensor, Tensor),
        y_support_emb: Tensor,
        y_query_mask: Tensor,
        n_s: usize,
        n_q: usize,
        n_feat: usize,
    ) -> Result<Tensor> {
        let (x_s_emb, x_q_emb) = x_emb;
        // Pack: y at column 0, features at columns 1..=n_feat.
        let mut support = Tensor::cat(&[&y_support_emb, &x_s_emb], 1)?; // (n_s, f+1, dim)
        let mut query = Tensor::cat(&[&y_query_mask, &x_q_emb], 1)?; // (n_q, f+1, dim)
        let f1 = n_feat + 1;

        for blk in &self.blocks {
            let (s2, q2) = query_block_forward(&support, &query, blk, self.config.n_heads, n_s, n_q, f1)?;
            support = s2;
            query = q2;
        }

        let query = zsfm_nn::layer_norm(&query, &self.norm_f_w, &self.norm_f_b, LN_EPS)?;
        let query = zsfm_nn::linear_bias(&query, &self.head_w, &self.head_b)?; // (n_q, f+1, dim_output)
        // Column 0 is the y-slot.
        query.narrow(1, 0, 1)?.reshape((n_q, self.config.dim_output))
            .map_err(anyhow::Error::from)
    }
}

/// One full `Layer` (row-attn → MLP → feature-attn → MLP) applied to both `support` and
/// `query` streams, mirroring `Tab2D`'s Python `Layer.forward` CPU path exactly.
fn query_block_forward(
    support: &Tensor,
    query: &Tensor,
    blk: &BlockW,
    n_heads: usize,
    n_s: usize,
    n_q: usize,
    f1: usize,
) -> Result<(Tensor, Tensor)> {
    // --- Row attention (across observations; query cross-attends to support) ---
    let res_s = support.clone();
    let res_q = query.clone();
    let s_ln = zsfm_nn::layer_norm(support, &blk.ln1_w, &blk.ln1_b, LN_EPS)?;
    let q_ln = zsfm_nn::layer_norm(query, &blk.ln1_w, &blk.ln1_b, LN_EPS)?;

    // (n, f, d) -> (f, n, d): attention batch = feature columns, sequence = observations.
    let s_row = s_ln.permute((1, 0, 2))?.contiguous()?;
    let q_row = q_ln.permute((1, 0, 2))?.contiguous()?;

    let s_att = mha(&s_row, &s_row, &s_row, &blk.attn_row, n_heads)?;
    let q_att = mha(&q_row, &s_row, &s_row, &blk.attn_row, n_heads)?;

    let s_att = s_att.permute((1, 0, 2))?.contiguous()?.reshape((n_s, f1, s_att.dim(2)?))?;
    let q_att = q_att.permute((1, 0, 2))?.contiguous()?.reshape((n_q, f1, q_att.dim(2)?))?;

    let mut support = (res_s + s_att)?;
    let mut query = (res_q + q_att)?;

    // --- MLP 1 ---
    let res_s = support.clone();
    let res_q = query.clone();
    let s_ln = zsfm_nn::layer_norm(&support, &blk.ln2_w, &blk.ln2_b, LN_EPS)?;
    let q_ln = zsfm_nn::layer_norm(&query, &blk.ln2_w, &blk.ln2_b, LN_EPS)?;
    let s_mlp = zsfm_nn::linear_bias(&zsfm_nn::linear_bias(&s_ln, &blk.mlp1_fc1_w, &blk.mlp1_fc1_b)?.gelu_erf()?, &blk.mlp1_fc2_w, &blk.mlp1_fc2_b)?;
    let q_mlp = zsfm_nn::linear_bias(&zsfm_nn::linear_bias(&q_ln, &blk.mlp1_fc1_w, &blk.mlp1_fc1_b)?.gelu_erf()?, &blk.mlp1_fc2_w, &blk.mlp1_fc2_b)?;
    support = (res_s + s_mlp)?;
    query = (res_q + q_mlp)?;

    // --- Feature attention (across columns, per observation; no cross term) ---
    let res_s = support.clone();
    let res_q = query.clone();
    let s_ln = zsfm_nn::layer_norm(&support, &blk.ln3_w, &blk.ln3_b, LN_EPS)?; // already (n_s, f1, d)
    let q_ln = zsfm_nn::layer_norm(&query, &blk.ln3_w, &blk.ln3_b, LN_EPS)?; // already (n_q, f1, d)

    let s_att = mha(&s_ln, &s_ln, &s_ln, &blk.attn_feat, n_heads)?;
    let q_att = mha(&q_ln, &q_ln, &q_ln, &blk.attn_feat, n_heads)?;

    support = (res_s + s_att)?;
    query = (res_q + q_att)?;

    // --- MLP 2 ---
    let res_s = support.clone();
    let res_q = query.clone();
    let s_ln = zsfm_nn::layer_norm(&support, &blk.ln4_w, &blk.ln4_b, LN_EPS)?;
    let q_ln = zsfm_nn::layer_norm(&query, &blk.ln4_w, &blk.ln4_b, LN_EPS)?;
    let s_mlp = zsfm_nn::linear_bias(&zsfm_nn::linear_bias(&s_ln, &blk.mlp2_fc1_w, &blk.mlp2_fc1_b)?.gelu_erf()?, &blk.mlp2_fc2_w, &blk.mlp2_fc2_b)?;
    let q_mlp = zsfm_nn::linear_bias(&zsfm_nn::linear_bias(&q_ln, &blk.mlp2_fc1_w, &blk.mlp2_fc1_b)?.gelu_erf()?, &blk.mlp2_fc2_w, &blk.mlp2_fc2_b)?;
    support = (res_s + s_mlp)?;
    query = (res_q + q_mlp)?;

    Ok((support, query))
}

/// Standard scaled-dot-product multi-head attention. `q`: `[batch, sq, dim]`, `k`/`v`:
/// `[batch, skv, dim]`. Uses the default PyTorch `scaled_dot_product_attention` scale of
/// `1/sqrt(head_dim)`.
fn mha(q: &Tensor, k: &Tensor, v: &Tensor, w: &AttnW, n_heads: usize) -> Result<Tensor> {
    let (batch, sq, dim) = q.dims3()?;
    let skv = k.dim(1)?;
    let head_dim = dim / n_heads;

    let q = zsfm_nn::linear_bias(q, &w.q_w, &w.q_b)?;
    let k = zsfm_nn::linear_bias(k, &w.k_w, &w.k_b)?;
    let v = zsfm_nn::linear_bias(v, &w.v_w, &w.v_b)?;

    let q = q.reshape((batch, sq, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
    let k = k.reshape((batch, skv, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
    let v = v.reshape((batch, skv, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

    let scale = (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = attn.matmul(&v)?; // (batch, h, sq, head_dim)
    let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch, sq, dim))?;
    zsfm_nn::linear_bias(&out, &w.o_w, &w.o_b)
}

// ---------------------------------------------------------------------------
// Preprocessing (Preprocessor.fit/transform_X, minus flags that default off:
// use_quantile_transformer, use_feature_count_scaling, use_random_transforms,
// shuffle_features, random_mirror_x)
// ---------------------------------------------------------------------------

/// Mean-impute NaNs (mean computed pre-imputation from `x_support`), then drop feature columns
/// that are constant across `x_support` (both computed from `x_support`, applied to both).
/// Returns `(x_support, x_query, kept_feature_indices)`.
fn preprocess_x(x_support: &[Vec<f32>], x_query: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    let n_feat = x_support[0].len();

    let mut col_mean = vec![0f32; n_feat];
    for f in 0..n_feat {
        let mut sum = 0f64;
        let mut count = 0usize;
        for row in x_support {
            if !row[f].is_nan() {
                sum += row[f] as f64;
                count += 1;
            }
        }
        col_mean[f] = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
    }

    let impute = |row: &[f32]| -> Vec<f32> {
        row.iter().enumerate().map(|(f, &v)| if v.is_nan() { col_mean[f] } else { v }).collect()
    };
    let support_imputed: Vec<Vec<f32>> = x_support.iter().map(|r| impute(r)).collect();
    let query_imputed: Vec<Vec<f32>> = x_query.iter().map(|r| impute(r)).collect();

    let kept: Vec<usize> = (0..n_feat)
        .filter(|&f| {
            let first = support_imputed[0][f];
            support_imputed.iter().any(|r| r[f] != first)
        })
        .collect();

    let select = |rows: &[Vec<f32>]| -> Vec<Vec<f32>> {
        rows.iter().map(|r| kept.iter().map(|&f| r[f]).collect()).collect()
    };

    (select(&support_imputed), select(&query_imputed), kept)
}

/// Per-feature quantile-bucketize normalization (`Tab2DQuantileEmbeddingX`): fit 999 quantile
/// boundaries per column from `x_support`, bucketize both `x_support`/`x_query` against them,
/// normalize the bucket index by `n_support`, then z-score using `x_support`'s own mean/std.
fn quantile_bucketize_normalize(x_support: &[Vec<f32>], x_query: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_support = x_support.len();
    let n_features = x_support[0].len();
    let n_query = x_query.len();

    let mut out_support = vec![vec![0f32; n_features]; n_support];
    let mut out_query = vec![vec![0f32; n_features]; n_query];

    for feat in 0..n_features {
        let mut col: Vec<f32> = x_support.iter().map(|r| r[feat]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let boundaries: Vec<f32> = (1..1000).map(|i| quantile_linear(&col, i as f64 / 1000.0)).collect();

        let bucket = |v: f32| -> f32 {
            let idx = boundaries.partition_point(|&b| b < v);
            idx as f32 / n_support as f32
        };

        let bucketed_support: Vec<f32> = x_support.iter().map(|r| bucket(r[feat])).collect();
        let mean: f32 = bucketed_support.iter().sum::<f32>() / n_support as f32;
        let var: f32 = bucketed_support.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n_support as f32;
        let std = var.sqrt();

        for (i, &v) in bucketed_support.iter().enumerate() {
            out_support[i][feat] = if std == 0.0 { 0.0 } else { (v - mean) / std };
        }
        for (i, row) in x_query.iter().enumerate() {
            let v = bucket(row[feat]);
            out_query[i][feat] = if std == 0.0 { 0.0 } else { (v - mean) / std };
        }
    }

    (out_support, out_query)
}

/// `torch.quantile` default ('linear') interpolation on an already-sorted slice.
fn quantile_linear(sorted: &[f32], q: f64) -> f32 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = (idx - lo as f64) as f32;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}
