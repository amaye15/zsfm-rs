//! TabFM inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle) single-table inference: Fourier cell
//! embedding, column SetTransformer stacks, row interaction with RoPE,
//! in-context transformer, decoder head. Weight loading goes through the
//! existing candle GGUF reader and converts each F32 tensor to Burn
//! `TensorData`, so there is exactly one GGUF parser.
//!
//! Ensemble orchestration (`ensemble::orchestrate`) is host-level code that
//! calls `predict_batch`; porting it is a matter of swapping the model type
//! once this single-table path passes its gate.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_rms_norm_nd;

use crate::config::TabFMConfig;

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

fn load_w(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
    expected_d_out: usize,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_weight(content, reader, name, expected_d_out, &CDevice::Cpu)?;
    burn2(&t)
}

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>, eps: f64) -> Tensor<D> {
    burn_rms_norm_nd(x, w, eps as f32)
}

fn sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    ((x.neg().exp() + 1.0).recip())
}

fn silu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.clone() * sigmoid(x)
}

fn softplus<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    (x.exp() + 1.0).log()
}

/// Candle `Tensor::gelu()` is the tanh approximation (see `Gelu` vs
/// `GeluErf` in candle-core `op.rs`), NOT erf. Match it exactly.
fn gelu_tanh<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::gelu_approximate(x)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

fn to_host_1(t: &Tensor<1>) -> Result<Vec<f32>> {
    to_host(t)
}

#[derive(Clone)]
struct BurnMabW {
    q_w: Tensor<2>,
    q_b: Tensor<1>,
    k_w: Tensor<2>,
    k_b: Tensor<1>,
    v_w: Tensor<2>,
    v_b: Tensor<1>,
    o_w: Tensor<2>,
    o_b: Tensor<1>,
    q_norm: Tensor<1>,
    k_norm: Tensor<1>,
    per_dim_scale: Tensor<1>,
    pre_attn_norm: Tensor<1>,
    post_attn_norm: Tensor<1>,
    pre_ff_norm: Tensor<1>,
    post_ff_norm: Tensor<1>,
    ffn_up_w: Tensor<2>,
    ffn_up_b: Tensor<1>,
    ffn_gate_w: Tensor<2>,
    ffn_gate_b: Tensor<1>,
    ffn_down_w: Tensor<2>,
    ffn_down_b: Tensor<1>,
}

struct BurnInducedBlockW {
    ind_vectors: Tensor<2>,
    mab1: BurnMabW,
    mab2: BurnMabW,
}

struct BurnColStackW {
    blocks: Vec<BurnInducedBlockW>,
    out_w_w: Tensor<2>,
    out_w_b: Tensor<1>,
    out_norm_w: Tensor<1>,
}

struct BurnRowStackW {
    rope_freqs: Vec<f32>,
    blocks: Vec<BurnMabW>,
    out_norm_w: Tensor<1>,
}

struct BurnMlpW {
    layers: Vec<(Tensor<2>, Tensor<1>)>,
}

enum BurnYEmbedW {
    Embedding(Tensor<2>),
    Mlp(BurnMlpW),
}

enum BurnYEncoderW {
    OneHot {
        proj_w: Tensor<2>,
        proj_b: Tensor<1>,
    },
    Mlp(BurnMlpW),
}

struct BurnCellW {
    fourier_freq: Tensor<2>,
    fourier_freq_cat: Tensor<2>,
    in_linear_w: Tensor<2>,
    in_linear_b: Tensor<1>,
    in_linear_cat_w: Tensor<2>,
    in_linear_cat_b: Tensor<1>,
    y_embed: BurnYEmbedW,
}

struct BurnIclW {
    blocks: Vec<BurnMabW>,
    out_norm_w: Tensor<1>,
    y_encoder: BurnYEncoderW,
    decoder: BurnMlpW,
}

pub struct BurnTabFMModel {
    config: BurnInferConfig,
    cell: BurnCellW,
    colenc1: BurnColStackW,
    colenc2: BurnColStackW,
    rowenc1: BurnRowStackW,
    rowenc2: BurnRowStackW,
    cls_tokens: Tensor<2>,
    icl: BurnIclW,
}

#[derive(Clone)]
struct BurnInferConfig {
    embed_dim: usize,
    max_classes: usize,
    col_nhead: usize,
    row_nhead: usize,
    row_num_cls: usize,
    icl_nhead: usize,
    ff_factor: usize,
    feature_group_size: usize,
    num_freq: usize,
    norm_eps: f64,
    is_classifier: bool,
}

fn load_mab(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    e: usize,
    ff: usize,
) -> Result<BurnMabW> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(BurnMabW {
        q_w: load_w(content, reader, &p("attn_q.weight"), e)?,
        q_b: load_v(content, reader, &p("attn_q.bias"))?,
        k_w: load_w(content, reader, &p("attn_k.weight"), e)?,
        k_b: load_v(content, reader, &p("attn_k.bias"))?,
        v_w: load_w(content, reader, &p("attn_v.weight"), e)?,
        v_b: load_v(content, reader, &p("attn_v.bias"))?,
        o_w: load_w(content, reader, &p("attn_o.weight"), e)?,
        o_b: load_v(content, reader, &p("attn_o.bias"))?,
        q_norm: load_v(content, reader, &p("q_norm.weight"))?,
        k_norm: load_v(content, reader, &p("k_norm.weight"))?,
        per_dim_scale: load_v(content, reader, &p("per_dim_scale"))?,
        pre_attn_norm: load_v(content, reader, &p("pre_attn_norm.weight"))?,
        post_attn_norm: load_v(content, reader, &p("post_attn_norm.weight"))?,
        pre_ff_norm: load_v(content, reader, &p("pre_ff_norm.weight"))?,
        post_ff_norm: load_v(content, reader, &p("post_ff_norm.weight"))?,
        ffn_up_w: load_w(content, reader, &p("ffn_up.weight"), ff)?,
        ffn_up_b: load_v(content, reader, &p("ffn_up.bias"))?,
        ffn_gate_w: load_w(content, reader, &p("ffn_gate.weight"), ff)?,
        ffn_gate_b: load_v(content, reader, &p("ffn_gate.bias"))?,
        ffn_down_w: load_w(content, reader, &p("ffn_down.weight"), e)?,
        ffn_down_b: load_v(content, reader, &p("ffn_down.bias"))?,
    })
}

fn load_col_stack(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    stack: &str,
    num_blocks: usize,
    e: usize,
    ff: usize,
) -> Result<BurnColStackW> {
    let mut blocks = Vec::with_capacity(num_blocks);
    for n in 0..num_blocks {
        let ind_vectors = load_t(content, reader, &format!("{stack}.blk.{n}.ind_vectors"))?;
        let mab1 = load_mab(content, reader, &format!("{stack}.blk.{n}.mab1"), e, ff)?;
        let mab2 = load_mab(content, reader, &format!("{stack}.blk.{n}.mab2"), e, ff)?;
        blocks.push(BurnInducedBlockW {
            ind_vectors,
            mab1,
            mab2,
        });
    }
    Ok(BurnColStackW {
        blocks,
        out_w_w: load_w(content, reader, &format!("{stack}.out_w.weight"), e)?,
        out_w_b: load_v(content, reader, &format!("{stack}.out_w.bias"))?,
        out_norm_w: load_v(content, reader, &format!("{stack}.out_norm.weight"))?,
    })
}

fn load_row_stack(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    stack: &str,
    num_blocks: usize,
    e: usize,
    ff: usize,
) -> Result<BurnRowStackW> {
    let rope_freqs: Vec<f32> = zsfm_nn::load_tensor(
        content,
        reader,
        &format!("{stack}.rope_freqs"),
        &CDevice::Cpu,
        DType::F32,
    )?
    .flatten_all()?
    .to_vec1()?;
    let mut blocks = Vec::with_capacity(num_blocks);
    for n in 0..num_blocks {
        blocks.push(load_mab(
            content,
            reader,
            &format!("{stack}.blk.{n}"),
            e,
            ff,
        )?);
    }
    Ok(BurnRowStackW {
        rope_freqs,
        blocks,
        out_norm_w: load_v(content, reader, &format!("{stack}.out_norm.weight"))?,
    })
}

fn load_mlp(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    dims: &[usize],
) -> Result<BurnMlpW> {
    let mut layers = Vec::with_capacity(dims.len() - 1);
    for i in 0..dims.len() - 1 {
        let out_dim = dims[i + 1];
        let w = {
            let t = zsfm_nn::load_weight(
                content,
                reader,
                &format!("{prefix}.mlp.{i}.weight"),
                out_dim,
                &CDevice::Cpu,
            )?;
            burn2(&t)?
        };
        let b = load_v(content, reader, &format!("{prefix}.mlp.{i}.bias"))?;
        layers.push((w, b));
    }
    Ok(BurnMlpW { layers })
}

struct BurnTabfmRope {
    freqs: Vec<f32>,
}

impl BurnTabfmRope {
    /// Rotate `x: [N, T, nhead, head_dim]` over its `T` (dim 1) axis with
    /// interleaved-pair frequencies from the checkpoint.
    fn rotate(&self, x: Tensor<4>) -> Result<Tensor<4>> {
        let xd = x.dims();
        let (n, t, nh, hd) = (xd[0], xd[1], xd[2], xd[3]);
        let dev = device();
        let half = hd / 2;
        let positions: Vec<f32> = (0..t).map(|p| p as f32).collect();
        let pos = Tensor::<1>::from_data(TensorData::new(positions, [t]), &dev);
        let freqs = Tensor::<1>::from_data(TensorData::new(self.freqs.clone(), [half]), &dev);
        let f = pos.unsqueeze_dim::<2>(1) * freqs.unsqueeze_dim::<2>(0);
        let cos = f.clone().cos();
        let sin = f.sin();
        let cos_i: Tensor<3> = Tensor::stack(vec![cos.clone(), cos], 2);
        let cos_i: Tensor<2> = cos_i.reshape([t, hd]);
        let sin_i: Tensor<3> = Tensor::stack(vec![sin.clone(), sin], 2);
        let sin_i: Tensor<2> = sin_i.reshape([t, hd]);
        let cos_i = cos_i.reshape([1, t, 1, hd]);
        let sin_i = sin_i.reshape([1, t, 1, hd]);
        let pairs = x.clone().reshape([n, t, nh, half, 2]);
        let x1 = pairs.clone().narrow(4, 0, 1).squeeze_dim::<4>(4);
        let x2 = pairs.narrow(4, 1, 1).squeeze_dim::<4>(4);
        let rot_stack: Tensor<5> = Tensor::stack(vec![x2.mul_scalar(-1.0), x1], 4);
        let rot = rot_stack.reshape([n, t, nh, hd]);
        Ok(x * cos_i + rot * sin_i)
    }
}
impl BurnTabFMModel {
    pub fn load(gguf_path: &Path, tc: &TabFMConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;
        let e = tc.embed_dim as usize;
        let col_ff = e * tc.ff_factor as usize;
        let icl_dim = e * tc.row_num_cls as usize;
        let icl_ff = icl_dim * tc.ff_factor as usize;
        let decoder_hidden = tc.decoder_hidden.map(|v| v as usize).unwrap_or(icl_dim * 2);
        let out_dim = if tc.is_classifier {
            tc.max_classes as usize
        } else {
            1
        };
        let config = BurnInferConfig {
            embed_dim: e,
            max_classes: tc.max_classes as usize,
            col_nhead: tc.col_nhead as usize,
            row_nhead: tc.row_nhead as usize,
            row_num_cls: tc.row_num_cls as usize,
            icl_nhead: tc.icl_nhead as usize,
            ff_factor: tc.ff_factor as usize,
            feature_group_size: tc.feature_group_size as usize,
            num_freq: tc.num_freq as usize,
            norm_eps: tc.norm_eps,
            is_classifier: tc.is_classifier,
        };
        let cell = BurnCellW {
            fourier_freq: load_t(&content, &mut reader, "cell.fourier_freq")?,
            fourier_freq_cat: load_t(&content, &mut reader, "cell.fourier_freq_cat")?,
            in_linear_w: load_w(&content, &mut reader, "cell.in_linear.weight", e)?,
            in_linear_b: load_v(&content, &mut reader, "cell.in_linear.bias")?,
            in_linear_cat_w: load_w(&content, &mut reader, "cell.in_linear_cat.weight", e)?,
            in_linear_cat_b: load_v(&content, &mut reader, "cell.in_linear_cat.bias")?,
            y_embed: if tc.is_classifier {
                BurnYEmbedW::Embedding(load_t(&content, &mut reader, "cell.y_embed.weight")?)
            } else {
                BurnYEmbedW::Mlp(load_mlp(&content, &mut reader, "cell.y_embed", &[1, 6, e])?)
            },
        };
        let colenc1 = load_col_stack(
            &content,
            &mut reader,
            "colenc1",
            tc.col_num_blocks as usize,
            e,
            col_ff,
        )?;
        let colenc2 = load_col_stack(
            &content,
            &mut reader,
            "colenc2",
            tc.col_num_blocks as usize,
            e,
            col_ff,
        )?;
        let rowenc1 = load_row_stack(
            &content,
            &mut reader,
            "rowenc1",
            tc.row_num_blocks as usize,
            e,
            col_ff,
        )?;
        let rowenc2 = load_row_stack(
            &content,
            &mut reader,
            "rowenc2",
            tc.row_num_blocks as usize,
            e,
            col_ff,
        )?;
        let cls_tokens = load_t(&content, &mut reader, "cls_tokens")?;
        let mut icl_blocks = Vec::with_capacity(tc.icl_num_blocks as usize);
        for n in 0..tc.icl_num_blocks as usize {
            icl_blocks.push(load_mab(
                &content,
                &mut reader,
                &format!("icl.blk.{n}"),
                icl_dim,
                icl_ff,
            )?);
        }
        let y_encoder = if tc.is_classifier {
            BurnYEncoderW::OneHot {
                proj_w: load_w(
                    &content,
                    &mut reader,
                    "icl.y_encoder.projection.weight",
                    icl_dim,
                )?,
                proj_b: load_v(&content, &mut reader, "icl.y_encoder.projection.bias")?,
            }
        } else {
            BurnYEncoderW::Mlp(load_mlp(
                &content,
                &mut reader,
                "icl.y_encoder",
                &[1, decoder_hidden, icl_dim],
            )?)
        };
        let decoder = load_mlp(
            &content,
            &mut reader,
            "icl.decoder",
            &[icl_dim, decoder_hidden, out_dim],
        )?;
        let icl = BurnIclW {
            blocks: icl_blocks,
            out_norm_w: load_v(&content, &mut reader, "icl.out_norm.weight")?,
            y_encoder,
            decoder,
        };
        Ok(Self {
            config,
            cell,
            colenc1,
            colenc2,
            rowenc1,
            rowenc2,
            cls_tokens,
            icl,
        })
    }

    pub fn is_classifier(&self) -> bool {
        self.config.is_classifier
    }

    pub fn predict(
        &self,
        x: &[Vec<f32>],
        y: &[f32],
        train_size: usize,
        cat_mask: Option<&[bool]>,
        d: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let out = self.predict_batch(
            &[x.to_vec()],
            &[y.to_vec()],
            train_size,
            &[cat_mask.unwrap_or(&[]).to_vec()],
            d,
        )?;
        out.into_iter()
            .next()
            .context("predict_batch returned no batch items")
    }

    pub fn predict_batch(
        &self,
        x_batch: &[Vec<Vec<f32>>],
        y_batch: &[Vec<f32>],
        train_size: usize,
        cat_mask_batch: &[Vec<bool>],
        d: Option<usize>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let cfg = &self.config;
        let b_len = x_batch.len();
        anyhow::ensure!(b_len > 0, "x_batch must have at least one table");
        let t_len = x_batch[0].len();
        let h_len = x_batch[0][0].len();
        let d_val = d.unwrap_or(h_len).min(h_len);
        let emb0 = self.cell_embed(x_batch, y_batch, train_size, cat_mask_batch, d_val)?;
        let emb1 = self.col_embedding_forward(&emb0, train_size, &self.colenc1)?;
        let num_cls = cfg.row_num_cls;
        let cls: Tensor<4> = self
            .cls_tokens
            .clone()
            .reshape([1, 1, num_cls, cfg.embed_dim])
            .expand([b_len, t_len, num_cls, cfg.embed_dim]);
        let emb2 = Tensor::cat(vec![cls, emb1], 2);
        let d_plus_cls = d_val + num_cls;
        let emb3 = self.row_interaction_forward(&emb2, d_plus_cls, &self.rowenc1)?;
        let emb4 = self.col_embedding_forward(&emb3, train_size, &self.colenc2)?;
        let reps = self.row_interaction_collapsed(&emb4, &self.rowenc2)?;
        let logits = self.icl_forward(&reps, y_batch, train_size)?;
        let out_dim = if cfg.is_classifier {
            cfg.max_classes
        } else {
            1
        };
        let flat: Vec<f32> = to_host(&logits)?;
        let mut result = vec![vec![vec![0f32; out_dim]; t_len]; b_len];
        for (bb, batch_item) in result.iter_mut().enumerate() {
            for (t, row) in batch_item.iter_mut().enumerate() {
                let base = ((bb * t_len) + t) * out_dim;
                row.copy_from_slice(&flat[base..base + out_dim]);
            }
        }
        Ok(result)
    }

    fn cell_embed(
        &self,
        x_batch: &[Vec<Vec<f32>>],
        y_batch: &[Vec<f32>],
        train_size: usize,
        cat_mask_batch: &[Vec<bool>],
        d: usize,
    ) -> Result<Tensor<4>> {
        let cfg = &self.config;
        let b_len = x_batch.len();
        let t_len = x_batch[0].len();
        let h_len = x_batch[0][0].len();
        let fgs = cfg.feature_group_size;
        let num_freq = cfg.num_freq;
        let e = cfg.embed_dim;
        let d_safe = d.max(1);
        let mut idx = vec![vec![0usize; h_len]; fgs];
        for (g, row) in idx.iter_mut().enumerate() {
            let offset = (1usize << g) - 1;
            for (h, slot) in row.iter_mut().enumerate() {
                *slot = (h + offset) % d_safe;
            }
        }
        let ff: Vec<f32> = to_host(&self.cell.fourier_freq)?;
        let ffc: Vec<f32> = to_host(&self.cell.fourier_freq_cat)?;
        let in_w: Vec<f32> = to_host(&self.cell.in_linear_w)?;
        let in_b: Vec<f32> = to_host_1(&self.cell.in_linear_b)?;
        let in_w_cat: Vec<f32> = to_host(&self.cell.in_linear_cat_w)?;
        let in_b_cat: Vec<f32> = to_host_1(&self.cell.in_linear_cat_b)?;
        let y_emb_t = compute_y_embed(y_batch, &self.cell.y_embed, cfg.max_classes)?;
        let y_emb: Vec<f32> = to_host(&y_emb_t)?;
        let mut out = vec![0f32; b_len * t_len * h_len * e];
        for bb in 0..b_len {
            let x = &x_batch[bb];
            let cat_mask = &cat_mask_batch[bb];
            for tt in 0..t_len {
                let add_y = tt < train_size;
                for hh in 0..h_len {
                    if hh >= d {
                        continue;
                    }
                    let mut acc = vec![0f32; e];
                    for g in 0..fgs {
                        let src_col = idx[g][hh];
                        let val = x[tt][src_col];
                        let is_cat = cat_mask.get(src_col).copied().unwrap_or(false);
                        let (freq_row, w_lin, b_lin): (&[f32], &[f32], &[f32]) = if is_cat {
                            (&ffc[g * num_freq..(g + 1) * num_freq], &in_w_cat, &in_b_cat)
                        } else {
                            (&ff[g * num_freq..(g + 1) * num_freq], &in_w, &in_b)
                        };
                        for ee in 0..e {
                            let mut s = b_lin[ee];
                            let row_off = ee * 2 * num_freq;
                            for f in 0..num_freq {
                                let arg = val * freq_row[f];
                                s += arg.sin() * w_lin[row_off + f];
                                s += arg.cos() * w_lin[row_off + num_freq + f];
                            }
                            acc[ee] += s;
                        }
                    }
                    let base = ((bb * t_len + tt) * h_len + hh) * e;
                    let y_base = (bb * t_len + tt) * e;
                    for ee in 0..e {
                        out[base + ee] = acc[ee] + if add_y { y_emb[y_base + ee] } else { 0.0 };
                    }
                }
            }
        }
        let dev = device();
        Ok(Tensor::<4>::from_data(
            TensorData::new(out, [b_len, t_len, h_len, e]),
            &dev,
        ))
    }

    fn col_embedding_forward(
        &self,
        x: &Tensor<4>,
        train_size: usize,
        w: &BurnColStackW,
    ) -> Result<Tensor<4>> {
        let cfg = &self.config;
        let xd = x.dims();
        let (b_len, t_len, hc, e) = (xd[0], xd[1], xd[2], xd[3]);
        let src: Tensor<3> = x
            .clone()
            .permute([0, 2, 1, 3])
            .reshape([b_len * hc, t_len, e]);
        let mask = additive_key_mask(t_len, train_size);
        let mut cur = src;
        for blk in &w.blocks {
            let n = cur.dims()[0];
            let ind: Tensor<3> = blk.ind_vectors.clone().unsqueeze_dim::<3>(0).expand([
                n,
                blk.ind_vectors.dims()[0],
                blk.ind_vectors.dims()[1],
            ]);
            let hidden = mab_forward(
                &ind,
                &cur,
                &cur,
                &blk.mab1,
                cfg.col_nhead,
                cfg.norm_eps,
                Some(mask.clone()),
                None,
            )?;
            cur = mab_forward(
                &cur,
                &hidden,
                &hidden,
                &blk.mab2,
                cfg.col_nhead,
                cfg.norm_eps,
                None,
                None,
            )?;
        }
        let projected = linear(cur, w.out_w_w.clone(), Some(w.out_w_b.clone()));
        let normed = rms(projected, w.out_norm_w.clone(), cfg.norm_eps);
        Ok(normed.reshape([b_len, hc, t_len, e]).permute([0, 2, 1, 3]))
    }

    fn row_interaction_forward(
        &self,
        x: &Tensor<4>,
        d_plus_cls: usize,
        w: &BurnRowStackW,
    ) -> Result<Tensor<4>> {
        let out3 = self.row_interaction_3d(x, d_plus_cls, w)?;
        let xd = x.dims();
        Ok(out3.reshape([xd[0], xd[1], xd[2], xd[3]]))
    }

    fn row_interaction_collapsed(&self, x: &Tensor<4>, w: &BurnRowStackW) -> Result<Tensor<3>> {
        let cfg = &self.config;
        let xd = x.dims();
        let (b_len, t_len) = (xd[0], xd[1]);
        let num_cls = cfg.row_num_cls;
        let out3 = self.row_interaction_3d(x, 0, w)?;
        let e = cfg.embed_dim;
        let sliced: Tensor<3> = out3.narrow(1, 0, num_cls);
        let normed = rms(sliced, w.out_norm_w.clone(), cfg.norm_eps);
        let icl_dim = num_cls * e;
        Ok(normed.reshape([b_len, t_len, icl_dim]))
    }

    fn row_interaction_3d(
        &self,
        x: &Tensor<4>,
        d_plus_cls: usize,
        w: &BurnRowStackW,
    ) -> Result<Tensor<3>> {
        let cfg = &self.config;
        let xd = x.dims();
        let (b_len, t_len, hc, e) = (xd[0], xd[1], xd[2], xd[3]);
        let mask = additive_key_mask(hc, d_plus_cls);
        let rope = BurnTabfmRope {
            freqs: w.rope_freqs.clone(),
        };
        let mut cur: Tensor<3> = x.clone().reshape([b_len * t_len, hc, e]);
        for blk in &w.blocks {
            cur = mab_forward(
                &cur,
                &cur,
                &cur,
                blk,
                cfg.row_nhead,
                cfg.norm_eps,
                Some(mask.clone()),
                Some(&rope),
            )?;
        }
        Ok(cur)
    }

    fn icl_forward(
        &self,
        reps: &Tensor<3>,
        y_batch: &[Vec<f32>],
        train_size: usize,
    ) -> Result<Tensor<3>> {
        let cfg = &self.config;
        let rd = reps.dims();
        let (b_len, t_len) = (rd[0], rd[1]);
        let y_enc = compute_y_encoder(y_batch, &self.icl.y_encoder, cfg.max_classes)?;
        let tm: Vec<f32> = (0..t_len)
            .map(|t| if t < train_size { 1.0 } else { 0.0 })
            .collect();
        let dev = device();
        let tm_t = Tensor::<2>::from_data(TensorData::new(tm, [t_len, 1]), &dev);
        let r = reps.clone() + y_enc * tm_t.unsqueeze_dim::<3>(0);
        let mask = additive_key_mask(t_len, train_size);
        let mut cur = r;
        for blk in &self.icl.blocks {
            cur = mab_forward(
                &cur,
                &cur,
                &cur,
                blk,
                cfg.icl_nhead,
                cfg.norm_eps,
                Some(mask.clone()),
                None,
            )?;
        }
        let normed = rms(cur, self.icl.out_norm_w.clone(), cfg.norm_eps);
        Ok(mlp_forward(&normed, &self.icl.decoder))
    }
}

fn compute_y_embed(y_batch: &[Vec<f32>], w: &BurnYEmbedW, max_classes: usize) -> Result<Tensor<3>> {
    let dev = device();
    let b_len = y_batch.len();
    let t = y_batch[0].len();
    match w {
        BurnYEmbedW::Embedding(emb_w) => {
            let e = emb_w.dims()[1];
            let emb: Vec<f32> = to_host(emb_w)?;
            let mut out = vec![0f32; b_len * t * e];
            for (bb, y) in y_batch.iter().enumerate() {
                for i in 0..t {
                    let yi = y[i] as i64;
                    if yi >= 0 && (yi as usize) < max_classes {
                        let cls = yi as usize;
                        let dst = (bb * t + i) * e;
                        out[dst..dst + e].copy_from_slice(&emb[cls * e..(cls + 1) * e]);
                    }
                }
            }
            Ok(Tensor::<3>::from_data(
                TensorData::new(out, [b_len, t, e]),
                &dev,
            ))
        }
        BurnYEmbedW::Mlp(mlp) => {
            let flat: Vec<f32> = y_batch.iter().flatten().copied().collect();
            let y_col = Tensor::<2>::from_data(TensorData::new(flat, [b_len * t, 1]), &dev);
            let out = mlp_forward(&y_col, mlp);
            let e = out.dims()[1];
            Ok(out.reshape([b_len, t, e]))
        }
    }
}

fn compute_y_encoder(
    y_batch: &[Vec<f32>],
    w: &BurnYEncoderW,
    max_classes: usize,
) -> Result<Tensor<3>> {
    let dev = device();
    let b_len = y_batch.len();
    let t = y_batch[0].len();
    match w {
        BurnYEncoderW::OneHot { proj_w, proj_b } => {
            let icl_dim = proj_w.dims()[0];
            let w_flat: Vec<f32> = to_host(proj_w)?;
            let bias: Vec<f32> = to_host_1(proj_b)?;
            let mut out = vec![0f32; b_len * t * icl_dim];
            for (bb, y) in y_batch.iter().enumerate() {
                for i in 0..t {
                    let yi = y[i] as i64;
                    let valid = yi >= 0 && (yi as usize) < max_classes;
                    let dst = (bb * t + i) * icl_dim;
                    for e in 0..icl_dim {
                        let w_contrib = if valid {
                            w_flat[e * max_classes + yi as usize]
                        } else {
                            0.0
                        };
                        out[dst + e] = bias[e] + w_contrib;
                    }
                }
            }
            Ok(Tensor::<3>::from_data(
                TensorData::new(out, [b_len, t, icl_dim]),
                &dev,
            ))
        }
        BurnYEncoderW::Mlp(mlp) => {
            let flat: Vec<f32> = y_batch.iter().flatten().copied().collect();
            let y_col = Tensor::<2>::from_data(TensorData::new(flat, [b_len * t, 1]), &dev);
            let out = mlp_forward(&y_col, mlp);
            let icl_dim = out.dims()[1];
            Ok(out.reshape([b_len, t, icl_dim]))
        }
    }
}

fn mlp_forward<const D: usize>(x: &Tensor<D>, w: &BurnMlpW) -> Tensor<D> {
    let n = w.layers.len();
    let mut h = x.clone();
    for (i, (lw, lb)) in w.layers.iter().enumerate() {
        h = linear(h, lw.clone(), Some(lb.clone()));
        if i < n - 1 {
            h = gelu_tanh(h);
        }
    }
    h
}

fn additive_key_mask(seq_len: usize, valid_len: usize) -> Tensor<4> {
    let dev = device();
    let data: Vec<f32> = (0..seq_len)
        .map(|i| if i < valid_len { 0.0 } else { -1e9 })
        .collect();
    Tensor::<1>::from_data(TensorData::new(data, [seq_len]), &dev).reshape([1, 1, 1, seq_len])
}

fn mab_forward(
    q_raw: &Tensor<3>,
    k_raw: &Tensor<3>,
    v_raw: &Tensor<3>,
    w: &BurnMabW,
    nhead: usize,
    eps: f64,
    mask: Option<Tensor<4>>,
    rope: Option<&BurnTabfmRope>,
) -> Result<Tensor<3>> {
    let qn = rms(q_raw.clone(), w.pre_attn_norm.clone(), eps);
    let kn = rms(k_raw.clone(), w.pre_attn_norm.clone(), eps);
    let vn = rms(v_raw.clone(), w.pre_attn_norm.clone(), eps);
    let attn_out = mha_core(&qn, &kn, &vn, w, nhead, eps, mask, rope)?;
    let a = rms(attn_out, w.post_attn_norm.clone(), eps);
    let x = q_raw.clone() + a;
    let xn = rms(x.clone(), w.pre_ff_norm.clone(), eps);
    let gate = silu(linear(
        xn.clone(),
        w.ffn_gate_w.clone(),
        Some(w.ffn_gate_b.clone()),
    ));
    let up = linear(xn, w.ffn_up_w.clone(), Some(w.ffn_up_b.clone()));
    let ff = linear(gate * up, w.ffn_down_w.clone(), Some(w.ffn_down_b.clone()));
    let ff = rms(ff, w.post_ff_norm.clone(), eps);
    Ok(x + ff)
}

fn mha_core(
    qn: &Tensor<3>,
    kn: &Tensor<3>,
    vn: &Tensor<3>,
    w: &BurnMabW,
    nhead: usize,
    eps: f64,
    mask: Option<Tensor<4>>,
    rope: Option<&BurnTabfmRope>,
) -> Result<Tensor<3>> {
    let qd = qn.dims();
    let (n, sq, e) = (qd[0], qd[1], qd[2]);
    let sk = kn.dims()[1];
    let hd = e / nhead;
    let q = linear(qn.clone(), w.q_w.clone(), Some(w.q_b.clone())).reshape([n, sq, nhead, hd]);
    let k = linear(kn.clone(), w.k_w.clone(), Some(w.k_b.clone())).reshape([n, sk, nhead, hd]);
    let v = linear(vn.clone(), w.v_w.clone(), Some(w.v_b.clone())).reshape([n, sk, nhead, hd]);
    let (q, k) = match rope {
        Some(r) => (r.rotate(q)?, r.rotate(k)?),
        None => (q, k),
    };
    let q = rms(q, w.q_norm.clone(), eps);
    let k = rms(k, w.k_norm.clone(), eps);
    let scale = softplus(w.per_dim_scale.clone()).mul_scalar(1.442695041f32 / (hd as f32).sqrt());
    let qd = q.dims();
    let q = (q.reshape([qd[0] * qd[1] * qd[2], hd]) * scale.unsqueeze_dim::<2>(0)).reshape(qd);
    let q: Tensor<4> = q.permute([0, 2, 1, 3]);
    let k: Tensor<4> = k.permute([0, 2, 1, 3]);
    let v: Tensor<4> = v.permute([0, 2, 1, 3]);
    let mut scores = q.matmul(k.transpose());
    if let Some(m) = mask {
        scores = scores + m;
    }
    let probs = softmax_last(scores);
    let out = probs.matmul(v);
    let out: Tensor<4> = out.permute([0, 2, 1, 3]);
    let out: Tensor<3> = out.reshape([n, sq, e]);
    Ok(linear(out, w.o_w.clone(), Some(w.o_b.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TabFMConfig;
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

    fn tiny_config_json() -> String {
        serde_json::json!({
            "embed_dim": 8,
            "max_classes": 2,
            "col_num_blocks": 1,
            "col_nhead": 2,
            "col_num_inds": 2,
            "row_num_blocks": 1,
            "row_nhead": 2,
            "row_num_cls": 2,
            "icl_num_blocks": 1,
            "icl_nhead": 2,
            "ff_factor": 2,
            "feature_group_size": 2,
            "is_classifier": true,
            "num_freq": 2
        })
        .to_string()
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

    fn put_mab(w: &mut GGUFWriter, prefix: &str, e: usize, ff: usize, seed: &mut u64, s: f32) {
        for t in ["attn_q", "attn_k", "attn_v", "attn_o"] {
            put2(w, &format!("{prefix}.{t}.weight"), e, e, seed, s);
            put1(w, &format!("{prefix}.{t}.bias"), e, seed);
        }
        put1(w, &format!("{prefix}.q_norm.weight"), e / 2, seed);
        put1(w, &format!("{prefix}.k_norm.weight"), e / 2, seed);
        put1(w, &format!("{prefix}.per_dim_scale"), e / 2, seed);
        for t in [
            "pre_attn_norm",
            "post_attn_norm",
            "pre_ff_norm",
            "post_ff_norm",
        ] {
            put1(w, &format!("{prefix}.{t}.weight"), e, seed);
        }
        put2(w, &format!("{prefix}.ffn_up.weight"), ff, e, seed, s);
        put1(w, &format!("{prefix}.ffn_up.bias"), ff, seed);
        put2(w, &format!("{prefix}.ffn_gate.weight"), ff, e, seed, s);
        put1(w, &format!("{prefix}.ffn_gate.bias"), ff, seed);
        put2(w, &format!("{prefix}.ffn_down.weight"), e, ff, seed, s);
        put1(w, &format!("{prefix}.ffn_down.bias"), e, seed);
    }

    #[test]
    fn burn_matches_candle_single_table() {
        let dir = std::env::temp_dir().join(format!("tabfm-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tabfm".into()),
        );
        let mut seed = 2500u64;
        let s = 0.05;
        put2(&mut w, "cell.fourier_freq", 2, 2, &mut seed, 1.0);
        put2(&mut w, "cell.fourier_freq_cat", 2, 2, &mut seed, 1.0);
        put2(&mut w, "cell.in_linear.weight", 8, 4, &mut seed, s);
        put1(&mut w, "cell.in_linear.bias", 8, &mut seed);
        put2(&mut w, "cell.in_linear_cat.weight", 8, 4, &mut seed, s);
        put1(&mut w, "cell.in_linear_cat.bias", 8, &mut seed);
        put2(&mut w, "cell.y_embed.weight", 2, 8, &mut seed, s);
        for stack in ["colenc1", "colenc2"] {
            put2(
                &mut w,
                &format!("{stack}.blk.0.ind_vectors"),
                2,
                8,
                &mut seed,
                s,
            );
            put_mab(&mut w, &format!("{stack}.blk.0.mab1"), 8, 16, &mut seed, s);
            put_mab(&mut w, &format!("{stack}.blk.0.mab2"), 8, 16, &mut seed, s);
            put2(&mut w, &format!("{stack}.out_w.weight"), 8, 8, &mut seed, s);
            put1(&mut w, &format!("{stack}.out_w.bias"), 8, &mut seed);
            put1(&mut w, &format!("{stack}.out_norm.weight"), 8, &mut seed);
        }
        for stack in ["rowenc1", "rowenc2"] {
            put1(&mut w, &format!("{stack}.rope_freqs"), 2, &mut seed);
            put_mab(&mut w, &format!("{stack}.blk.0"), 8, 16, &mut seed, s);
            put1(&mut w, &format!("{stack}.out_norm.weight"), 8, &mut seed);
        }
        put2(&mut w, "cls_tokens", 2, 8, &mut seed, s);
        put_mab(&mut w, "icl.blk.0", 16, 32, &mut seed, s);
        put2(
            &mut w,
            "icl.y_encoder.projection.weight",
            16,
            2,
            &mut seed,
            s,
        );
        put1(&mut w, "icl.y_encoder.projection.bias", 16, &mut seed);
        put2(&mut w, "icl.decoder.mlp.0.weight", 32, 16, &mut seed, s);
        put1(&mut w, "icl.decoder.mlp.0.bias", 32, &mut seed);
        put2(&mut w, "icl.decoder.mlp.1.weight", 2, 32, &mut seed, s);
        put1(&mut w, "icl.decoder.mlp.1.bias", 2, &mut seed);
        put1(&mut w, "icl.out_norm.weight", 16, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let tc = TabFMConfig::from_json(&tiny_config_json()).unwrap();
        let candle = crate::TabFMModel::builder(&path)
            .config_from(&tc)
            .build()
            .unwrap();
        let burn = BurnTabFMModel::load(&path, &tc).unwrap();
        let x = vec![
            vec![0.1, 1.2],
            vec![0.9, -0.3],
            vec![-1.1, 0.4],
            vec![1.5, 1.1],
            vec![0.2, 0.9],
            vec![-0.8, 0.1],
        ];
        let y = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let t0 = std::time::Instant::now();
        let a = candle.predict(&x, &y, 4, None, None).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.predict(&x, &y, 4, None, None).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let mut err = 0.0f32;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (u, v) in ra.iter().zip(rb.iter()) {
                err = err.max((u - v).abs());
            }
        }
        println!("tabfm synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-3, "tabfm Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
