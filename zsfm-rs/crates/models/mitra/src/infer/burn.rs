//! Mitra inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): quantile-bucketize embedding,
//! row/feature dual attention blocks, gated head readout. Weight loading
//! goes through the existing candle GGUF reader and converts each F32
//! tensor to Burn `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_layer_norm_nd;

use crate::config::{MitraConfig, Task};

const LN_EPS: f32 = 1e-5;

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

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn layer_norm<const D: usize>(x: Tensor<D>, w: Tensor<1>, b: Tensor<1>) -> Tensor<D> {
    burn_layer_norm_nd(x, w, b, LN_EPS)
}

fn gelu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::gelu(x)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

struct BurnAttnW {
    q_w: Tensor<2>,
    q_b: Tensor<1>,
    k_w: Tensor<2>,
    k_b: Tensor<1>,
    v_w: Tensor<2>,
    v_b: Tensor<1>,
    o_w: Tensor<2>,
    o_b: Tensor<1>,
}

struct BurnBlockW {
    ln1_w: Tensor<1>,
    ln1_b: Tensor<1>,
    ln2_w: Tensor<1>,
    ln2_b: Tensor<1>,
    ln3_w: Tensor<1>,
    ln3_b: Tensor<1>,
    ln4_w: Tensor<1>,
    ln4_b: Tensor<1>,
    attn_row: BurnAttnW,
    attn_feat: BurnAttnW,
    mlp1_fc1_w: Tensor<2>,
    mlp1_fc1_b: Tensor<1>,
    mlp1_fc2_w: Tensor<2>,
    mlp1_fc2_b: Tensor<1>,
    mlp2_fc1_w: Tensor<2>,
    mlp2_fc1_b: Tensor<1>,
    mlp2_fc2_w: Tensor<2>,
    mlp2_fc2_b: Tensor<1>,
}

pub struct BurnMitraModel {
    config: MitraConfig,
    x_embed_w: Tensor<2>,
    x_embed_b: Tensor<1>,
    y_embed_w: Tensor<2>,
    y_embed_b: Option<Tensor<1>>,
    y_mask_w: Tensor<1>,
    blocks: Vec<BurnBlockW>,
    norm_f_w: Tensor<1>,
    norm_f_b: Tensor<1>,
    head_w: Tensor<2>,
    head_b: Tensor<1>,
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnAttnW> {
    Ok(BurnAttnW {
        q_w: load_t(content, reader, &format!("{prefix}.q.weight"))?,
        q_b: load_v(content, reader, &format!("{prefix}.q.bias"))?,
        k_w: load_t(content, reader, &format!("{prefix}.k.weight"))?,
        k_b: load_v(content, reader, &format!("{prefix}.k.bias"))?,
        v_w: load_t(content, reader, &format!("{prefix}.v.weight"))?,
        v_b: load_v(content, reader, &format!("{prefix}.v.bias"))?,
        o_w: load_t(content, reader, &format!("{prefix}.o.weight"))?,
        o_b: load_v(content, reader, &format!("{prefix}.o.bias"))?,
    })
}

impl BurnMitraModel {
    pub fn load(gguf_path: &Path, config: MitraConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let x_embed_w = load_t(&content, &mut reader, "x_embed.weight")?;
        let x_embed_b = load_v(&content, &mut reader, "x_embed.bias")?;
        let y_embed_w = load_t(&content, &mut reader, "y_embed.weight")?;
        let y_embed_b = match config.task {
            Task::Regression => Some(load_v(&content, &mut reader, "y_embed.bias")?),
            Task::Classification => None,
        };
        let y_mask_w = load_v(&content, &mut reader, "y_mask.weight")?;

        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            blocks.push(BurnBlockW {
                ln1_w: load_v(&content, &mut reader, &p("ln1.weight"))?,
                ln1_b: load_v(&content, &mut reader, &p("ln1.bias"))?,
                ln2_w: load_v(&content, &mut reader, &p("ln2.weight"))?,
                ln2_b: load_v(&content, &mut reader, &p("ln2.bias"))?,
                ln3_w: load_v(&content, &mut reader, &p("ln3.weight"))?,
                ln3_b: load_v(&content, &mut reader, &p("ln3.bias"))?,
                ln4_w: load_v(&content, &mut reader, &p("ln4.weight"))?,
                ln4_b: load_v(&content, &mut reader, &p("ln4.bias"))?,
                attn_row: load_attn(&content, &mut reader, &p("attn_row"))?,
                attn_feat: load_attn(&content, &mut reader, &p("attn_feat"))?,
                mlp1_fc1_w: load_t(&content, &mut reader, &p("mlp1_fc1.weight"))?,
                mlp1_fc1_b: load_v(&content, &mut reader, &p("mlp1_fc1.bias"))?,
                mlp1_fc2_w: load_t(&content, &mut reader, &p("mlp1_fc2.weight"))?,
                mlp1_fc2_b: load_v(&content, &mut reader, &p("mlp1_fc2.bias"))?,
                mlp2_fc1_w: load_t(&content, &mut reader, &p("mlp2_fc1.weight"))?,
                mlp2_fc1_b: load_v(&content, &mut reader, &p("mlp2_fc1.bias"))?,
                mlp2_fc2_w: load_t(&content, &mut reader, &p("mlp2_fc2.weight"))?,
                mlp2_fc2_b: load_v(&content, &mut reader, &p("mlp2_fc2.bias"))?,
            });
        }

        Ok(Self {
            config,
            x_embed_w,
            x_embed_b,
            y_embed_w,
            y_embed_b,
            y_mask_w,
            blocks,
            norm_f_w: load_v(&content, &mut reader, "norm_f.weight")?,
            norm_f_b: load_v(&content, &mut reader, "norm_f.bias")?,
            head_w: load_t(&content, &mut reader, "head.weight")?,
            head_b: load_v(&content, &mut reader, "head.bias")?,
        })
    }

    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(
            self.config.task == Task::Classification,
            "model was loaded as a regressor"
        );
        let (x_s, x_q, kept) = preprocess_x(x_support, x_query);
        let n_s = x_s.len();
        let n_q = x_q.len();
        let n_feat = kept.len();
        let x_emb = self.embed_x(&x_s, &x_q, n_s, n_q, n_feat);
        let y_support_emb = self.embed_y_classes_support(y_support, n_s)?;
        let y_query_mask = self.embed_y_mask(n_q);
        let logits = self.forward(x_emb, y_support_emb, y_query_mask, n_s, n_q, n_feat);
        let logits: Vec<f32> = to_host(&logits)?;
        let dim_out = self.config.dim_output;
        Ok((0..n_q)
            .map(|i| logits[i * dim_out..i * dim_out + n_classes].to_vec())
            .collect())
    }

    pub fn predict_regression(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[f32],
        x_query: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.config.task == Task::Regression,
            "model was loaded as a classifier"
        );
        let (x_s, x_q, kept) = preprocess_x(x_support, x_query);
        let n_s = x_s.len();
        let n_q = x_q.len();
        let n_feat = kept.len();
        let y_min = y_support.iter().copied().fold(f32::INFINITY, f32::min);
        let y_max = y_support.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        anyhow::ensure!(
            y_max > y_min,
            "y_support must have at least two distinct values"
        );
        let y_scaled: Vec<f32> = y_support
            .iter()
            .map(|&v| (v - y_min) / (y_max - y_min))
            .collect();
        let x_emb = self.embed_x(&x_s, &x_q, n_s, n_q, n_feat);
        let y_support_emb = self.embed_y_regression_support(&y_scaled)?;
        let y_query_mask = self.embed_y_mask(n_q);
        let out = self.forward(x_emb, y_support_emb, y_query_mask, n_s, n_q, n_feat);
        let out: Vec<f32> = to_host(&out)?;
        Ok(out
            .into_iter()
            .map(|v| v * (y_max - y_min) + y_min)
            .collect())
    }

    fn embed_x(
        &self,
        x_s: &[Vec<f32>],
        x_q: &[Vec<f32>],
        n_s: usize,
        n_q: usize,
        n_feat: usize,
    ) -> (Tensor<3>, Tensor<3>) {
        let dev = device();
        let (bx_s, bx_q) = quantile_bucketize_normalize(x_s, x_q);
        let flat_s: Vec<f32> = bx_s.into_iter().flatten().collect();
        let flat_q: Vec<f32> = bx_q.into_iter().flatten().collect();
        let d = self.config.dim;
        let x_s_t = Tensor::<2>::from_data(TensorData::new(flat_s, [n_s * n_feat, 1]), &dev);
        let x_q_t = Tensor::<2>::from_data(TensorData::new(flat_q, [n_q * n_feat, 1]), &dev);
        let x_s_emb = linear(x_s_t, self.x_embed_w.clone(), Some(self.x_embed_b.clone()))
            .reshape([n_s, n_feat, d]);
        let x_q_emb = linear(x_q_t, self.x_embed_w.clone(), Some(self.x_embed_b.clone()))
            .reshape([n_q, n_feat, d]);
        (x_s_emb, x_q_emb)
    }

    fn embed_y_classes_support(&self, y_support: &[usize], n_s: usize) -> Result<Tensor<3>> {
        let dim = self.config.dim;
        let table: Vec<f32> = to_host(&self.y_embed_w)?;
        let mut out = vec![0f32; n_s * dim];
        for (i, &cls) in y_support.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&table[cls * dim..(cls + 1) * dim]);
        }
        Ok(Tensor::<3>::from_data(
            TensorData::new(out, [n_s, 1, dim]),
            &device(),
        ))
    }

    fn embed_y_regression_support(&self, y_scaled: &[f32]) -> Result<Tensor<3>> {
        let n_s = y_scaled.len();
        let dev = device();
        let y_t = Tensor::<2>::from_data(TensorData::new(y_scaled.to_vec(), [n_s, 1]), &dev);
        let b = self
            .y_embed_b
            .as_ref()
            .context("regressor y_embed missing bias")?;
        let emb = linear(y_t, self.y_embed_w.clone(), Some(b.clone()));
        Ok(emb.reshape([n_s, 1, self.config.dim]))
    }

    fn embed_y_mask(&self, n_q: usize) -> Tensor<3> {
        let dim = self.config.dim;
        let row: Vec<f32> = to_host_1(&self.y_mask_w).expect("mask read");
        let mut out = Vec::with_capacity(n_q * dim);
        for _ in 0..n_q {
            out.extend_from_slice(&row);
        }
        Tensor::<3>::from_data(TensorData::new(out, [n_q, 1, dim]), &device())
    }

    fn forward(
        &self,
        x_emb: (Tensor<3>, Tensor<3>),
        y_support_emb: Tensor<3>,
        y_query_mask: Tensor<3>,
        n_s: usize,
        n_q: usize,
        n_feat: usize,
    ) -> Tensor<2> {
        let (x_s_emb, x_q_emb) = x_emb;
        let mut support = Tensor::cat(vec![y_support_emb, x_s_emb], 1);
        let mut query = Tensor::cat(vec![y_query_mask, x_q_emb], 1);
        let f1 = n_feat + 1;
        for blk in &self.blocks {
            let (s2, q2) =
                query_block_forward(&support, &query, blk, self.config.n_heads, n_s, n_q, f1);
            support = s2;
            query = q2;
        }
        let query = layer_norm(query, self.norm_f_w.clone(), self.norm_f_b.clone());
        let query = linear(query, self.head_w.clone(), Some(self.head_b.clone()));
        query.narrow(1, 0, 1).reshape([n_q, self.config.dim_output])
    }
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn to_host_1(t: &Tensor<1>) -> Result<Vec<f32>> {
    to_host(t)
}

fn query_block_forward(
    support: &Tensor<3>,
    query: &Tensor<3>,
    blk: &BurnBlockW,
    n_heads: usize,
    n_s: usize,
    n_q: usize,
    f1: usize,
) -> (Tensor<3>, Tensor<3>) {
    let s_ln = layer_norm(support.clone(), blk.ln1_w.clone(), blk.ln1_b.clone());
    let q_ln = layer_norm(query.clone(), blk.ln1_w.clone(), blk.ln1_b.clone());
    let s_row: Tensor<3> = s_ln.permute([1, 0, 2]);
    let q_row: Tensor<3> = q_ln.permute([1, 0, 2]);
    let s_att = mha(&s_row, &s_row, &s_row, &blk.attn_row, n_heads);
    let q_att = mha(&q_row, &s_row, &s_row, &blk.attn_row, n_heads);
    let s_att: Tensor<3> = s_att.permute([1, 0, 2]);
    let q_att: Tensor<3> = q_att.permute([1, 0, 2]);
    let s_dim = s_att.dims()[2];
    let q_dim = q_att.dims()[2];
    let s_att = s_att.reshape([n_s, f1, s_dim]);
    let q_att = q_att.reshape([n_q, f1, q_dim]);
    let mut support = support.clone() + s_att;
    let mut query = query.clone() + q_att;

    let s_ln = layer_norm(support.clone(), blk.ln2_w.clone(), blk.ln2_b.clone());
    let q_ln = layer_norm(query.clone(), blk.ln2_w.clone(), blk.ln2_b.clone());
    let s_mlp = linear(
        gelu(linear(
            s_ln,
            blk.mlp1_fc1_w.clone(),
            Some(blk.mlp1_fc1_b.clone()),
        )),
        blk.mlp1_fc2_w.clone(),
        Some(blk.mlp1_fc2_b.clone()),
    );
    let q_mlp = linear(
        gelu(linear(
            q_ln,
            blk.mlp1_fc1_w.clone(),
            Some(blk.mlp1_fc1_b.clone()),
        )),
        blk.mlp1_fc2_w.clone(),
        Some(blk.mlp1_fc2_b.clone()),
    );
    support = support + s_mlp;
    query = query + q_mlp;

    let s_ln = layer_norm(support.clone(), blk.ln3_w.clone(), blk.ln3_b.clone());
    let q_ln = layer_norm(query.clone(), blk.ln3_w.clone(), blk.ln3_b.clone());
    let s_att = mha(&s_ln, &s_ln, &s_ln, &blk.attn_feat, n_heads);
    let q_att = mha(&q_ln, &q_ln, &q_ln, &blk.attn_feat, n_heads);
    support = support + s_att;
    query = query + q_att;

    let s_ln = layer_norm(support.clone(), blk.ln4_w.clone(), blk.ln4_b.clone());
    let q_ln = layer_norm(query.clone(), blk.ln4_w.clone(), blk.ln4_b.clone());
    let s_mlp = linear(
        gelu(linear(
            s_ln,
            blk.mlp2_fc1_w.clone(),
            Some(blk.mlp2_fc1_b.clone()),
        )),
        blk.mlp2_fc2_w.clone(),
        Some(blk.mlp2_fc2_b.clone()),
    );
    let q_mlp = linear(
        gelu(linear(
            q_ln,
            blk.mlp2_fc1_w.clone(),
            Some(blk.mlp2_fc1_b.clone()),
        )),
        blk.mlp2_fc2_w.clone(),
        Some(blk.mlp2_fc2_b.clone()),
    );
    support = support + s_mlp;
    query = query + q_mlp;
    (support, query)
}

fn mha(q: &Tensor<3>, k: &Tensor<3>, v: &Tensor<3>, w: &BurnAttnW, n_heads: usize) -> Tensor<3> {
    let qd = q.dims();
    let (batch, sq, dim) = (qd[0], qd[1], qd[2]);
    let kd = k.dims();
    let skv = kd[1];
    let head_dim = dim / n_heads;
    let q = linear(q.clone(), w.q_w.clone(), Some(w.q_b.clone()));
    let k = linear(k.clone(), w.k_w.clone(), Some(w.k_b.clone()));
    let v = linear(v.clone(), w.v_w.clone(), Some(w.v_b.clone()));
    let q: Tensor<4> = q
        .reshape([batch, sq, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let k: Tensor<4> = k
        .reshape([batch, skv, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let v: Tensor<4> = v
        .reshape([batch, skv, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let attn = activation::softmax(scores, 3);
    let out = attn.matmul(v);
    let out: Tensor<4> = out.permute([0, 2, 1, 3]);
    let out: Tensor<3> = out.reshape([batch, sq, dim]);
    linear(out, w.o_w.clone(), Some(w.o_b.clone()))
}

fn preprocess_x(
    x_support: &[Vec<f32>],
    x_query: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
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
        col_mean[f] = if count > 0 {
            (sum / count as f64) as f32
        } else {
            0.0
        };
    }
    let impute = |row: &[f32]| -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(f, &v)| if v.is_nan() { col_mean[f] } else { v })
            .collect()
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
        rows.iter()
            .map(|r| kept.iter().map(|&f| r[f]).collect())
            .collect()
    };
    (select(&support_imputed), select(&query_imputed), kept)
}

fn quantile_bucketize_normalize(
    x_support: &[Vec<f32>],
    x_query: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_support = x_support.len();
    let n_features = x_support[0].len();
    let n_query = x_query.len();
    let mut out_support = vec![vec![0f32; n_features]; n_support];
    let mut out_query = vec![vec![0f32; n_features]; n_query];
    for feat in 0..n_features {
        let mut col: Vec<f32> = x_support.iter().map(|r| r[feat]).collect();
        col.sort_by(|a, b| a.total_cmp(b));
        let boundaries: Vec<f32> = (1..1000)
            .map(|i| quantile_linear(&col, i as f64 / 1000.0))
            .collect();
        let bucket = |v: f32| -> f32 {
            let idx = boundaries.partition_point(|&b| b < v);
            idx as f32 / n_support as f32
        };
        let bucketed_support: Vec<f32> = x_support.iter().map(|r| bucket(r[feat])).collect();
        let mean: f32 = bucketed_support.iter().sum::<f32>() / n_support as f32;
        let var: f32 = bucketed_support
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f32>()
            / n_support as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Task as CfgTask;
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

    fn tiny_config() -> MitraConfig {
        MitraConfig {
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            dim_output: 2,
            task: CfgTask::Classification,
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

    fn put_attn(w: &mut GGUFWriter, prefix: &str, d: usize, seed: &mut u64, s: f32) {
        for t in ["q", "k", "v", "o"] {
            put2(w, &format!("{prefix}.{t}.weight"), d, d, seed, s);
            put1(w, &format!("{prefix}.{t}.bias"), d, seed);
        }
    }

    fn put_block(w: &mut GGUFWriter, n: usize, d: usize, seed: &mut u64, s: f32) {
        for ln in ["ln1", "ln2", "ln3", "ln4"] {
            put1(w, &format!("blk.{n}.{ln}.weight"), d, seed);
            put1(w, &format!("blk.{n}.{ln}.bias"), d, seed);
        }
        put_attn(w, &format!("blk.{n}.attn_row"), d, seed, s);
        put_attn(w, &format!("blk.{n}.attn_feat"), d, seed, s);
        for m in ["mlp1", "mlp2"] {
            put2(w, &format!("blk.{n}.{m}_fc1.weight"), 2 * d, d, seed, s);
            put1(w, &format!("blk.{n}.{m}_fc1.bias"), 2 * d, seed);
            put2(w, &format!("blk.{n}.{m}_fc2.weight"), d, 2 * d, seed, s);
            put1(w, &format!("blk.{n}.{m}_fc2.bias"), d, seed);
        }
    }

    #[test]
    fn burn_matches_candle_classification() {
        let dir = std::env::temp_dir().join(format!("mitra-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("mitra".into()),
        );
        let mut seed = 1700u64;
        let s = 0.2;
        put2(&mut w, "x_embed.weight", 8, 1, &mut seed, s);
        put1(&mut w, "x_embed.bias", 8, &mut seed);
        put2(&mut w, "y_embed.weight", 2, 8, &mut seed, s);
        put1(&mut w, "y_mask.weight", 8, &mut seed);
        put_block(&mut w, 0, 8, &mut seed, s);
        put1(&mut w, "norm_f.weight", 8, &mut seed);
        put1(&mut w, "norm_f.bias", 8, &mut seed);
        put2(&mut w, "head.weight", 2, 8, &mut seed, s);
        put1(&mut w, "head.bias", 2, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::MitraModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnMitraModel::load(&path, cfg).unwrap();
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
        println!("mitra synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "mitra Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
