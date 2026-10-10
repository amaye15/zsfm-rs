//! TabICL v2 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): column Set-Transformer embedding,
//! row interaction with RoPE, in-context transformer with SSMax, decoder
//! head. Weight loading goes through the existing candle GGUF reader and
//! converts each F32 tensor to Burn `TensorData`, so there is exactly one
//! GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_layer_norm_nd;
use zsfm_burn::rope::apply_llama_rope;

use crate::config::TabIclConfig;

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
    zsfm_burn::linear::burn_linear_nd(x, w, b)
}

fn linear_nb<const D: usize>(x: Tensor<D>, w: Tensor<2>) -> Tensor<D> {
    zsfm_burn::linear::burn_linear_nd(x, w, None)
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

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnSsmaxW {
    base_fc1_w: Tensor<2>,
    base_fc1_b: Tensor<1>,
    base_fc2_w: Tensor<2>,
    base_fc2_b: Tensor<1>,
    query_fc1_w: Tensor<2>,
    query_fc1_b: Tensor<1>,
    query_fc2_w: Tensor<2>,
    query_fc2_b: Tensor<1>,
}

struct BurnAttnW {
    in_proj_w: Tensor<2>,
    in_proj_b: Tensor<1>,
    out_proj_w: Tensor<2>,
    out_proj_b: Tensor<1>,
    ssmax: Option<BurnSsmaxW>,
}

struct BurnBlockW {
    norm1_w: Tensor<1>,
    norm1_b: Tensor<1>,
    norm2_w: Tensor<1>,
    norm2_b: Tensor<1>,
    attn: BurnAttnW,
    fc1_w: Tensor<2>,
    fc1_b: Tensor<1>,
    fc2_w: Tensor<2>,
    fc2_b: Tensor<1>,
}

struct BurnIsabW {
    ind_vectors: Tensor<2>,
    attn1: BurnBlockW,
    attn2: BurnBlockW,
}

struct BurnColW {
    in_linear_w: Tensor<2>,
    in_linear_b: Tensor<1>,
    y_encoder_w: Tensor<2>,
    y_encoder_b: Tensor<1>,
    blocks: Vec<BurnIsabW>,
}

struct BurnRowW {
    cls_tokens: Tensor<2>,
    blocks: Vec<BurnBlockW>,
    out_ln_w: Tensor<1>,
    out_ln_b: Tensor<1>,
    rope_freqs: Vec<f32>,
}

struct BurnIclW {
    y_encoder_w: Tensor<2>,
    y_encoder_b: Tensor<1>,
    blocks: Vec<BurnBlockW>,
    ln_w: Tensor<1>,
    ln_b: Tensor<1>,
    decoder_fc1_w: Tensor<2>,
    decoder_fc1_b: Tensor<1>,
    decoder_fc2_w: Tensor<2>,
    decoder_fc2_b: Tensor<1>,
}

pub struct BurnTabIclModel {
    config: TabIclConfig,
    col: BurnColW,
    row: BurnRowW,
    icl: BurnIclW,
}

fn load_ssmax(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnSsmaxW> {
    Ok(BurnSsmaxW {
        base_fc1_w: load_t(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.base_mlp.0.weight"),
        )?,
        base_fc1_b: load_v(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.base_mlp.0.bias"),
        )?,
        base_fc2_w: load_t(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.base_mlp.2.weight"),
        )?,
        base_fc2_b: load_v(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.base_mlp.2.bias"),
        )?,
        query_fc1_w: load_t(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.query_mlp.0.weight"),
        )?,
        query_fc1_b: load_v(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.query_mlp.0.bias"),
        )?,
        query_fc2_w: load_t(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.query_mlp.2.weight"),
        )?,
        query_fc2_b: load_v(
            content,
            reader,
            &format!("{prefix}.ssmax_layer.query_mlp.2.bias"),
        )?,
    })
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    has_ssmax: bool,
) -> Result<BurnAttnW> {
    Ok(BurnAttnW {
        in_proj_w: load_t(content, reader, &format!("{prefix}.in_proj_weight"))?,
        in_proj_b: load_v(content, reader, &format!("{prefix}.in_proj_bias"))?,
        out_proj_w: load_t(content, reader, &format!("{prefix}.out_proj.weight"))?,
        out_proj_b: load_v(content, reader, &format!("{prefix}.out_proj.bias"))?,
        ssmax: if has_ssmax {
            Some(load_ssmax(content, reader, prefix)?)
        } else {
            None
        },
    })
}

fn load_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    has_ssmax: bool,
) -> Result<BurnBlockW> {
    Ok(BurnBlockW {
        norm1_w: load_v(content, reader, &format!("{prefix}.norm1.weight"))?,
        norm1_b: load_v(content, reader, &format!("{prefix}.norm1.bias"))?,
        norm2_w: load_v(content, reader, &format!("{prefix}.norm2.weight"))?,
        norm2_b: load_v(content, reader, &format!("{prefix}.norm2.bias"))?,
        attn: load_attn(content, reader, &format!("{prefix}.attn"), has_ssmax)?,
        fc1_w: load_t(content, reader, &format!("{prefix}.linear1.weight"))?,
        fc1_b: load_v(content, reader, &format!("{prefix}.linear1.bias"))?,
        fc2_w: load_t(content, reader, &format!("{prefix}.linear2.weight"))?,
        fc2_b: load_v(content, reader, &format!("{prefix}.linear2.bias"))?,
    })
}

impl BurnTabIclModel {
    pub fn load(gguf_path: &Path, config: TabIclConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let col_in_linear_w = load_t(&content, &mut reader, "col_embedder.in_linear.weight")?;
        let col_in_linear_b = load_v(&content, &mut reader, "col_embedder.in_linear.bias")?;
        let col_y_w = load_t(&content, &mut reader, "col_embedder.y_encoder.weight")?;
        let col_y_b = load_v(&content, &mut reader, "col_embedder.y_encoder.bias")?;

        let mut col_blocks = Vec::with_capacity(config.col_num_blocks);
        for n in 0..config.col_num_blocks {
            let p = format!("col_embedder.tf_col.blocks.{n}");
            let ind_vectors = load_t(&content, &mut reader, &format!("{p}.ind_vectors"))?;
            let attn1 = load_block(&content, &mut reader, &format!("{p}.multihead_attn1"), true)?;
            let attn2 = load_block(
                &content,
                &mut reader,
                &format!("{p}.multihead_attn2"),
                false,
            )?;
            col_blocks.push(BurnIsabW {
                ind_vectors,
                attn1,
                attn2,
            });
        }

        let cls_tokens = load_t(&content, &mut reader, "row_interactor.cls_tokens")?;
        let row_out_ln_w = load_v(&content, &mut reader, "row_interactor.out_ln.weight")?;
        let row_out_ln_b = load_v(&content, &mut reader, "row_interactor.out_ln.bias")?;
        let rope_freqs: Vec<f32> = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "row_interactor.tf_row.rope.freqs",
            &CDevice::Cpu,
            DType::F32,
        )?
        .flatten_all()?
        .to_vec1()?;
        let mut row_blocks = Vec::with_capacity(config.row_num_blocks);
        for n in 0..config.row_num_blocks {
            row_blocks.push(load_block(
                &content,
                &mut reader,
                &format!("row_interactor.tf_row.blocks.{n}"),
                false,
            )?);
        }

        let mut icl_blocks = Vec::with_capacity(config.icl_num_blocks);
        for n in 0..config.icl_num_blocks {
            icl_blocks.push(load_block(
                &content,
                &mut reader,
                &format!("icl_predictor.tf_icl.blocks.{n}"),
                true,
            )?);
        }

        Ok(Self {
            config,
            col: BurnColW {
                in_linear_w: col_in_linear_w,
                in_linear_b: col_in_linear_b,
                y_encoder_w: col_y_w,
                y_encoder_b: col_y_b,
                blocks: col_blocks,
            },
            row: BurnRowW {
                cls_tokens,
                blocks: row_blocks,
                out_ln_w: row_out_ln_w,
                out_ln_b: row_out_ln_b,
                rope_freqs,
            },
            icl: BurnIclW {
                y_encoder_w: load_t(&content, &mut reader, "icl_predictor.y_encoder.weight")?,
                y_encoder_b: load_v(&content, &mut reader, "icl_predictor.y_encoder.bias")?,
                blocks: icl_blocks,
                ln_w: load_v(&content, &mut reader, "icl_predictor.ln.weight")?,
                ln_b: load_v(&content, &mut reader, "icl_predictor.ln.bias")?,
                decoder_fc1_w: load_t(&content, &mut reader, "icl_predictor.decoder.0.weight")?,
                decoder_fc1_b: load_v(&content, &mut reader, "icl_predictor.decoder.0.bias")?,
                decoder_fc2_w: load_t(&content, &mut reader, "icl_predictor.decoder.2.weight")?,
                decoder_fc2_b: load_v(&content, &mut reader, "icl_predictor.decoder.2.bias")?,
            },
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
            n_classes <= self.config.max_classes,
            "n_classes must be <= {}",
            self.config.max_classes
        );
        let train_size = x_support.len();
        let mut all_rows = x_support.to_vec();
        all_rows.extend_from_slice(x_query);
        let t = all_rows.len();
        let h = all_rows[0].len();
        let processed = preprocess_x(&all_rows, train_size);
        let group_size = self.config.feature_group_size;
        let grouped = feature_group_same(&processed, h, group_size);
        let mut flat = vec![0f32; h * t * group_size];
        for (row_idx, row) in grouped.iter().enumerate() {
            for (col_idx, group) in row.iter().enumerate() {
                let base = col_idx * t * group_size + row_idx * group_size;
                flat[base..base + group_size].copy_from_slice(group);
            }
        }
        let dev = device();
        let x_t = Tensor::<3>::from_data(TensorData::new(flat, [h, t, group_size]), &dev);
        let y_onehot_col =
            self.onehot_linear(y_support, &self.col.y_encoder_w, &self.col.y_encoder_b)?;
        let col_out = self.col_embed_forward(&x_t, &y_onehot_col, train_size)?;
        let row_out = self.row_interact_forward(&col_out, train_size)?;
        let y_onehot_icl =
            self.onehot_linear(y_support, &self.icl.y_encoder_w, &self.icl.y_encoder_b)?;
        let logits = self.icl_forward(&row_out, &y_onehot_icl, train_size, n_classes)?;
        let flat_logits: Vec<f32> = to_host(&logits)?;
        let n_query = t - train_size;
        const TEMPERATURE: f32 = 0.9;
        let mut result = Vec::with_capacity(n_query);
        for i in 0..n_query {
            let row = &flat_logits[i * n_classes..(i + 1) * n_classes];
            let scaled: Vec<f32> = row.iter().map(|&v| v / TEMPERATURE).collect();
            result.push(zsfm_core::softmax(&scaled));
        }
        Ok(result)
    }

    fn onehot_linear(&self, y: &[usize], w: &Tensor<2>, b: &Tensor<1>) -> Result<Tensor<2>> {
        let dev = device();
        let n = y.len();
        let num_classes = w.dims()[1];
        let mut onehot = vec![0f32; n * num_classes];
        for (i, &c) in y.iter().enumerate() {
            onehot[i * num_classes + c] = 1.0;
        }
        let x = Tensor::<2>::from_data(TensorData::new(onehot, [n, num_classes]), &dev);
        Ok(linear(x, w.clone(), Some(b.clone())))
    }

    fn col_embed_forward(
        &self,
        x_grouped: &Tensor<3>,
        y_onehot: &Tensor<2>,
        train_size: usize,
    ) -> Result<Tensor<3>> {
        let dims = x_grouped.dims();
        let t = dims[1];
        let src = linear(
            x_grouped.clone(),
            self.col.in_linear_w.clone(),
            Some(self.col.in_linear_b.clone()),
        );
        let train_part: Tensor<3> = src.clone().narrow(1, 0, train_size);
        // y_onehot [train, dim] -> [1, train, dim] broadcast over H.
        let y4: Tensor<3> = y_onehot.clone().unsqueeze_dim::<3>(0);
        let h = src.dims()[0];
        let yd = y4.dims()[2];
        let yb: Tensor<3> = y4.expand([h, train_size, yd]);
        let train_biased: Tensor<3> = train_part + yb;
        let mut src = if train_size < t {
            Tensor::cat(
                vec![train_biased, src.narrow(1, train_size, t - train_size)],
                1,
            )
        } else {
            train_biased
        };
        for isab in &self.col.blocks {
            src = isab_forward(&src, isab, self.config.col_nhead, train_size, h)?;
        }
        Ok(src)
    }

    fn row_interact_forward(
        &self,
        col_embeddings: &Tensor<3>,
        _train_size: usize,
    ) -> Result<Tensor<2>> {
        let dev = device();
        let cd = col_embeddings.dims();
        let (h, t) = (cd[0], cd[1]);
        let dim = self.config.embed_dim;
        let n_cls = self.config.row_num_cls;
        let feat: Tensor<3> = col_embeddings.clone().permute([1, 0, 2]);
        let cls: Tensor<3> = self
            .row
            .cls_tokens
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([t, n_cls, dim]);
        let mut seq = Tensor::cat(vec![cls, feat], 1);
        let max_len = h + n_cls;
        let (cos, sin) = self.row_rope_table(max_len)?;
        let n_blocks = self.row.blocks.len();
        for (i, blk) in self.row.blocks.iter().enumerate() {
            if i + 1 == n_blocks {
                let cls_q: Tensor<3> = seq.clone().narrow(1, 0, n_cls);
                seq = mha_block(
                    &cls_q,
                    &seq,
                    &seq,
                    blk,
                    self.config.row_nhead,
                    Some((cos.clone(), sin.clone())),
                    0,
                )?;
            } else {
                seq = mha_block(
                    &seq,
                    &seq,
                    &seq,
                    blk,
                    self.config.row_nhead,
                    Some((cos.clone(), sin.clone())),
                    0,
                )?;
            }
        }
        let _ = dev;
        let out = layer_norm(seq, self.row.out_ln_w.clone(), self.row.out_ln_b.clone());
        Ok(out.reshape([t, n_cls * dim]))
    }

    fn row_rope_table(&self, max_len: usize) -> Result<(Tensor<2>, Tensor<2>)> {
        let dev = device();
        let half = self.row.rope_freqs.len();
        let mut cos = vec![0f32; max_len * half * 2];
        let mut sin = vec![0f32; max_len * half * 2];
        for p in 0..max_len {
            for i in 0..half {
                let angle = p as f32 * self.row.rope_freqs[i];
                let (s, c) = angle.sin_cos();
                cos[p * 2 * half + i] = c;
                cos[p * 2 * half + half + i] = c;
                sin[p * 2 * half + i] = s;
                sin[p * 2 * half + half + i] = s;
            }
        }
        Ok((
            Tensor::<2>::from_data(TensorData::new(cos, [max_len, 2 * half]), &dev),
            Tensor::<2>::from_data(TensorData::new(sin, [max_len, 2 * half]), &dev),
        ))
    }

    fn icl_forward(
        &self,
        row_reprs: &Tensor<2>,
        y_onehot: &Tensor<2>,
        train_size: usize,
        n_classes: usize,
    ) -> Result<Tensor<2>> {
        let t = row_reprs.dims()[0];
        let train_part: Tensor<2> = row_reprs.clone().narrow(0, 0, train_size);
        let train_biased = train_part + y_onehot.clone();
        let r = if train_size < t {
            Tensor::cat(
                vec![
                    train_biased,
                    row_reprs.clone().narrow(0, train_size, t - train_size),
                ],
                0,
            )
        } else {
            train_biased
        };
        let mut r: Tensor<3> = r.unsqueeze_dim::<3>(0);
        for blk in &self.icl.blocks {
            let kv: Tensor<3> = r.clone().narrow(1, 0, train_size);
            r = mha_block(&r, &kv, &kv, blk, self.config.icl_nhead, None, train_size)?;
        }
        let r: Tensor<2> = r.squeeze_dim::<2>(0);
        let r = layer_norm(r, self.icl.ln_w.clone(), self.icl.ln_b.clone());
        let h = gelu(linear(
            r,
            self.icl.decoder_fc1_w.clone(),
            Some(self.icl.decoder_fc1_b.clone()),
        ));
        let logits = linear(
            h,
            self.icl.decoder_fc2_w.clone(),
            Some(self.icl.decoder_fc2_b.clone()),
        );
        Ok(logits
            .narrow(0, train_size, t - train_size)
            .narrow(1, 0, n_classes))
    }
}

fn isab_forward(
    src: &Tensor<3>,
    isab: &BurnIsabW,
    n_heads: usize,
    train_size: usize,
    n_h: usize,
) -> Result<Tensor<3>> {
    let sd = src.dims();
    let (num_inds, dim) = (isab.ind_vectors.dims()[0], isab.ind_vectors.dims()[1]);
    let ind: Tensor<3> = isab
        .ind_vectors
        .clone()
        .unsqueeze_dim::<3>(0)
        .expand([n_h, num_inds, dim]);
    let kv_train: Tensor<3> = src.clone().narrow(1, 0, train_size);
    let hidden = mha_block(
        &ind,
        &kv_train,
        &kv_train,
        &isab.attn1,
        n_heads,
        None,
        train_size,
    )?;
    Ok(mha_block(
        src,
        &hidden,
        &hidden,
        &isab.attn2,
        n_heads,
        None,
        num_inds,
    )?)
}

fn mha_block(
    q_in: &Tensor<3>,
    k_in: &Tensor<3>,
    v_in: &Tensor<3>,
    blk: &BurnBlockW,
    n_heads: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
    ssmax_n: usize,
) -> Result<Tensor<3>> {
    let q_normed = layer_norm(q_in.clone(), blk.norm1_w.clone(), blk.norm1_b.clone());
    let k_normed = layer_norm(k_in.clone(), blk.norm1_w.clone(), blk.norm1_b.clone());
    let v_normed = layer_norm(v_in.clone(), blk.norm1_w.clone(), blk.norm1_b.clone());
    let attn_out = mha_combined(
        &q_normed,
        &k_normed,
        &v_normed,
        &blk.attn,
        n_heads,
        rope_cos_sin,
        ssmax_n,
    )?;
    let x = q_in.clone() + attn_out;
    let ff_in = layer_norm(x.clone(), blk.norm2_w.clone(), blk.norm2_b.clone());
    let ff = linear(ff_in, blk.fc1_w.clone(), Some(blk.fc1_b.clone()));
    let ff = linear(gelu(ff), blk.fc2_w.clone(), Some(blk.fc2_b.clone()));
    Ok(x + ff)
}

fn mha_combined(
    q_in: &Tensor<3>,
    k_in: &Tensor<3>,
    v_in: &Tensor<3>,
    attn: &BurnAttnW,
    n_heads: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
    ssmax_n: usize,
) -> Result<Tensor<3>> {
    let qd = q_in.dims();
    let dim = qd[2];
    let wq: Tensor<2> = attn.in_proj_w.clone().narrow(0, 0, dim);
    let wk: Tensor<2> = attn.in_proj_w.clone().narrow(0, dim, dim);
    let wv: Tensor<2> = attn.in_proj_w.clone().narrow(0, 2 * dim, dim);
    let bq: Tensor<1> = attn.in_proj_b.clone().narrow(0, 0, dim);
    let bk: Tensor<1> = attn.in_proj_b.clone().narrow(0, dim, dim);
    let bv: Tensor<1> = attn.in_proj_b.clone().narrow(0, 2 * dim, dim);
    let q = linear(q_in.clone(), wq, Some(bq));
    let k = linear(k_in.clone(), wk, Some(bk));
    let v = linear(v_in.clone(), wv, Some(bv));
    let batch = q.dims()[0];
    let q_len = q.dims()[1];
    let k_len = k.dims()[1];
    let head_dim = dim / n_heads;
    let q: Tensor<4> = q
        .reshape([batch, q_len, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let k: Tensor<4> = k
        .reshape([batch, k_len, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let v: Tensor<4> = v
        .reshape([batch, k_len, n_heads, head_dim])
        .permute([0, 2, 1, 3]);
    let (q, k) = match rope_cos_sin {
        Some((cos, sin)) => (
            apply_llama_rope(q, cos.clone(), sin.clone(), 0),
            apply_llama_rope(k, cos, sin, 0),
        ),
        None => (q, k),
    };
    let q = match &attn.ssmax {
        Some(ssmax) => apply_ssmax(q, ssmax, ssmax_n, n_heads, head_dim)?,
        None => q,
    };
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scores = q.matmul(k.transpose()).mul_scalar(scale);
    let probs = activation::softmax(scores, 3);
    let out = probs.matmul(v);
    let out: Tensor<4> = out.permute([0, 2, 1, 3]);
    let out: Tensor<3> = out.reshape([batch, q_len, dim]);
    Ok(linear(
        out,
        attn.out_proj_w.clone(),
        Some(attn.out_proj_b.clone()),
    ))
}

fn apply_ssmax(
    q: Tensor<4>,
    ssmax: &BurnSsmaxW,
    n: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor<4>> {
    let dev = device();
    let logn = (n.max(1) as f32).ln();
    let logn_t = Tensor::<2>::from_data(TensorData::new(vec![logn], [1, 1]), &dev);
    let base = gelu(linear(
        logn_t,
        ssmax.base_fc1_w.clone(),
        Some(ssmax.base_fc1_b.clone()),
    ));
    let base = linear(
        base,
        ssmax.base_fc2_w.clone(),
        Some(ssmax.base_fc2_b.clone()),
    );
    let base: Tensor<4> = base.reshape([1, n_heads, 1, head_dim]);
    let qd = q.dims();
    let (batch, seq) = (qd[0], qd[2]);
    let q_flat: Tensor<2> = q.clone().reshape([batch * n_heads * seq, head_dim]);
    let qm = gelu(linear(
        q_flat,
        ssmax.query_fc1_w.clone(),
        Some(ssmax.query_fc1_b.clone()),
    ));
    let qm = linear(
        qm,
        ssmax.query_fc2_w.clone(),
        Some(ssmax.query_fc2_b.clone()),
    );
    let modulation: Tensor<4> = qm
        .tanh()
        .add_scalar(1.0)
        .reshape([batch, n_heads, seq, head_dim]);
    let scales = base * modulation;
    Ok(q * scales)
}

fn preprocess_x(rows: &[Vec<f32>], train_size: usize) -> Vec<Vec<f32>> {
    let n_feat = rows[0].len();
    let mut impute_mean = vec![0f32; n_feat];
    for f in 0..n_feat {
        let mut sum = 0f64;
        let mut count = 0usize;
        for row in &rows[..train_size] {
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
    let imputed: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(f, &v)| if v.is_nan() { impute_mean[f] } else { v })
                .collect()
        })
        .collect();
    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let m: f32 = imputed[..train_size].iter().map(|r| r[f]).sum::<f32>() / train_size as f32;
        let var: f32 = imputed[..train_size]
            .iter()
            .map(|r| (r[f] - m).powi(2))
            .sum::<f32>()
            / train_size as f32;
        mean[f] = m;
        std[f] = if var == 0.0 { 1.0 } else { var.sqrt() };
    }
    imputed
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(f, &v)| (v - mean[f]) / std[f])
                .collect()
        })
        .collect()
}

fn feature_group_same(rows: &[Vec<f32>], h: usize, size: usize) -> Vec<Vec<Vec<f32>>> {
    rows.iter()
        .map(|row| {
            (0..h)
                .map(|g| (0..size).map(|k| row[(g + (1usize << k)) % h]).collect())
                .collect()
        })
        .collect()
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

    fn tiny_config() -> TabIclConfig {
        TabIclConfig {
            max_classes: 2,
            embed_dim: 8,
            col_num_blocks: 1,
            col_nhead: 2,
            col_num_inds: 2,
            feature_group_size: 2,
            row_num_blocks: 1,
            row_nhead: 2,
            row_num_cls: 2,
            row_rope_base: 10000.0,
            icl_num_blocks: 1,
            icl_nhead: 2,
            ff_factor: 2,
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

    fn put_attn_ssmax(
        w: &mut GGUFWriter,
        prefix: &str,
        d: usize,
        seed: &mut u64,
        s: f32,
        ssmax_hd: Option<usize>,
        ssmax_base_out: usize,
    ) {
        put2(w, &format!("{prefix}.in_proj_weight"), 3 * d, d, seed, s);
        put1(w, &format!("{prefix}.in_proj_bias"), 3 * d, seed);
        put2(w, &format!("{prefix}.out_proj.weight"), d, d, seed, s);
        put1(w, &format!("{prefix}.out_proj.bias"), d, seed);
        if let Some(hd) = ssmax_hd {
            put2(
                w,
                &format!("{prefix}.ssmax_layer.base_mlp.0.weight"),
                8,
                1,
                seed,
                s,
            );
            put1(w, &format!("{prefix}.ssmax_layer.base_mlp.0.bias"), 8, seed);
            put2(
                w,
                &format!("{prefix}.ssmax_layer.base_mlp.2.weight"),
                ssmax_base_out,
                8,
                seed,
                s,
            );
            put1(
                w,
                &format!("{prefix}.ssmax_layer.base_mlp.2.bias"),
                ssmax_base_out,
                seed,
            );
            put2(
                w,
                &format!("{prefix}.ssmax_layer.query_mlp.0.weight"),
                8,
                hd,
                seed,
                s,
            );
            put1(
                w,
                &format!("{prefix}.ssmax_layer.query_mlp.0.bias"),
                8,
                seed,
            );
            put2(
                w,
                &format!("{prefix}.ssmax_layer.query_mlp.2.weight"),
                hd,
                8,
                seed,
                s,
            );
            put1(
                w,
                &format!("{prefix}.ssmax_layer.query_mlp.2.bias"),
                hd,
                seed,
            );
        }
    }

    fn put_block(
        w: &mut GGUFWriter,
        prefix: &str,
        d: usize,
        seed: &mut u64,
        s: f32,
        ssmax: Option<(usize, usize)>,
    ) {
        put1(w, &format!("{prefix}.norm1.weight"), d, seed);
        put1(w, &format!("{prefix}.norm1.bias"), d, seed);
        put1(w, &format!("{prefix}.norm2.weight"), d, seed);
        put1(w, &format!("{prefix}.norm2.bias"), d, seed);
        let (hd, base_out) = ssmax.map(|(a, b)| (Some(a), b)).unwrap_or((None, d));
        put_attn_ssmax(w, &format!("{prefix}.attn"), d, seed, s, hd, base_out);
        put2(w, &format!("{prefix}.linear1.weight"), 2 * d, d, seed, s);
        put1(w, &format!("{prefix}.linear1.bias"), 2 * d, seed);
        put2(w, &format!("{prefix}.linear2.weight"), d, 2 * d, seed, s);
        put1(w, &format!("{prefix}.linear2.bias"), d, seed);
    }

    #[test]
    fn burn_matches_candle_classification() {
        let dir = std::env::temp_dir().join(format!("tabicl-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tabicl".into()),
        );
        let mut seed = 2100u64;
        let s = 0.2;
        put2(&mut w, "col_embedder.in_linear.weight", 8, 2, &mut seed, s);
        put1(&mut w, "col_embedder.in_linear.bias", 8, &mut seed);
        put2(&mut w, "col_embedder.y_encoder.weight", 8, 2, &mut seed, s);
        put1(&mut w, "col_embedder.y_encoder.bias", 8, &mut seed);
        put2(
            &mut w,
            "col_embedder.tf_col.blocks.0.ind_vectors",
            2,
            8,
            &mut seed,
            s,
        );
        put_block(
            &mut w,
            "col_embedder.tf_col.blocks.0.multihead_attn1",
            8,
            &mut seed,
            s,
            Some((4, 8)),
        );
        put_block(
            &mut w,
            "col_embedder.tf_col.blocks.0.multihead_attn2",
            8,
            &mut seed,
            s,
            None,
        );
        put2(&mut w, "row_interactor.cls_tokens", 2, 8, &mut seed, s);
        put1(&mut w, "row_interactor.out_ln.weight", 8, &mut seed);
        put1(&mut w, "row_interactor.out_ln.bias", 8, &mut seed);
        put1(&mut w, "row_interactor.tf_row.rope.freqs", 2, &mut seed);
        put_block(
            &mut w,
            "row_interactor.tf_row.blocks.0",
            8,
            &mut seed,
            s,
            None,
        );
        put2(
            &mut w,
            "icl_predictor.y_encoder.weight",
            16,
            2,
            &mut seed,
            s,
        );
        put1(&mut w, "icl_predictor.y_encoder.bias", 16, &mut seed);
        put1(&mut w, "icl_predictor.ln.weight", 16, &mut seed);
        put1(&mut w, "icl_predictor.ln.bias", 16, &mut seed);
        put2(
            &mut w,
            "icl_predictor.decoder.0.weight",
            8,
            16,
            &mut seed,
            s,
        );
        put1(&mut w, "icl_predictor.decoder.0.bias", 8, &mut seed);
        put2(&mut w, "icl_predictor.decoder.2.weight", 2, 8, &mut seed, s);
        put1(&mut w, "icl_predictor.decoder.2.bias", 2, &mut seed);
        put_block(
            &mut w,
            "icl_predictor.tf_icl.blocks.0",
            16,
            &mut seed,
            s,
            Some((8, 16)),
        );
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::TabIclModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnTabIclModel::load(&path, cfg).unwrap();
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
        println!("tabicl synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "tabicl Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
