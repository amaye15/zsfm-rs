//! TabPFN-3 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): feature distribution embedder,
//! column aggregator with RoPE, ICL transformer with test-only GQA head
//! reduction, ManyClassDecoder readout. Weight loading goes through the
//! existing candle GGUF reader and converts each F32 tensor to Burn
//! `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_rms_norm_nd;

use crate::config::TabPfnConfig;

const RMS_EPS: f32 = f32::EPSILON;

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

fn linear_nb<const D: usize>(x: Tensor<D>, w: Tensor<2>) -> Tensor<D> {
    burn_linear_nd(x, w, None)
}

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>) -> Tensor<D> {
    burn_rms_norm_nd(x, w, RMS_EPS)
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

struct BurnQkvW {
    q_w: Tensor<2>,
    k_w: Tensor<2>,
    v_w: Tensor<2>,
    out_w: Tensor<2>,
    ssmax: Option<BurnSsmaxW>,
}

struct BurnMlpW {
    fc1_w: Tensor<2>,
    fc2_w: Tensor<2>,
}

struct BurnCrossAttnBlockW {
    attn: BurnQkvW,
    mlp: BurnMlpW,
    ln_q_w: Tensor<1>,
    ln_kv_w: Tensor<1>,
    ln2_w: Tensor<1>,
}

struct BurnIsabW {
    ind_vectors: Tensor<2>,
    block1: BurnCrossAttnBlockW,
    block2: BurnCrossAttnBlockW,
}

struct BurnTransformerBlockW {
    attn: BurnQkvW,
    mlp: BurnMlpW,
    ln_w: Tensor<1>,
    ln_mlp_w: Tensor<1>,
}

struct BurnIclBlockW {
    attn: BurnQkvW,
    mlp: BurnMlpW,
    ln_w: Tensor<1>,
    ln_mlp_w: Tensor<1>,
}

struct BurnManyClassDecoderW {
    q_w: Tensor<2>,
    q_b: Tensor<1>,
    k_w: Tensor<2>,
    k_b: Tensor<1>,
    ssmax: Option<BurnSsmaxW>,
}

pub struct BurnTabPfnModel {
    config: TabPfnConfig,
    x_embed_w: Tensor<2>,
    x_embed_b: Tensor<1>,
    col_y_encoder_w: Tensor<2>,
    icl_y_encoder_w: Tensor<2>,
    dist_embed: Vec<BurnIsabW>,
    col_agg_blocks: Vec<BurnTransformerBlockW>,
    col_agg_cls_tokens: Tensor<2>,
    col_agg_rope_freqs: Vec<f32>,
    col_agg_out_ln_w: Tensor<1>,
    icl_blocks: Vec<BurnIclBlockW>,
    output_norm_w: Tensor<1>,
    many_class_decoder: BurnManyClassDecoderW,
}

fn load_ssmax(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnSsmaxW> {
    Ok(BurnSsmaxW {
        base_fc1_w: load_t(content, reader, &format!("{prefix}.base_mlp.0.weight"))?,
        base_fc1_b: load_v(content, reader, &format!("{prefix}.base_mlp.0.bias"))?,
        base_fc2_w: load_t(content, reader, &format!("{prefix}.base_mlp.2.weight"))?,
        base_fc2_b: load_v(content, reader, &format!("{prefix}.base_mlp.2.bias"))?,
        query_fc1_w: load_t(content, reader, &format!("{prefix}.query_mlp.0.weight"))?,
        query_fc1_b: load_v(content, reader, &format!("{prefix}.query_mlp.0.bias"))?,
        query_fc2_w: load_t(content, reader, &format!("{prefix}.query_mlp.2.weight"))?,
        query_fc2_b: load_v(content, reader, &format!("{prefix}.query_mlp.2.bias"))?,
    })
}

fn load_qkv(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    ssmax_prefix: Option<&str>,
) -> Result<BurnQkvW> {
    Ok(BurnQkvW {
        q_w: load_t(content, reader, &format!("{prefix}.q_projection.weight"))?,
        k_w: load_t(content, reader, &format!("{prefix}.k_projection.weight"))?,
        v_w: load_t(content, reader, &format!("{prefix}.v_projection.weight"))?,
        out_w: load_t(content, reader, &format!("{prefix}.out_projection.weight"))?,
        ssmax: match ssmax_prefix {
            Some(p) => Some(load_ssmax(content, reader, p)?),
            None => None,
        },
    })
}

fn load_mlp(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnMlpW> {
    Ok(BurnMlpW {
        fc1_w: load_t(content, reader, &format!("{prefix}.0.weight"))?,
        fc2_w: load_t(content, reader, &format!("{prefix}.2.weight"))?,
    })
}

fn load_cross_attn_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    has_ssmax: bool,
) -> Result<BurnCrossAttnBlockW> {
    let ssmax_prefix = format!("{prefix}.attn.softmax_scaling_layer");
    Ok(BurnCrossAttnBlockW {
        attn: load_qkv(
            content,
            reader,
            &format!("{prefix}.attn"),
            has_ssmax.then_some(ssmax_prefix.as_str()),
        )?,
        mlp: load_mlp(content, reader, &format!("{prefix}.mlp"))?,
        ln_q_w: load_v(content, reader, &format!("{prefix}.layernorm_q.weight"))?,
        ln_kv_w: load_v(content, reader, &format!("{prefix}.layernorm_kv.weight"))?,
        ln2_w: load_v(content, reader, &format!("{prefix}.layernorm2.weight"))?,
    })
}

fn load_transformer_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnTransformerBlockW> {
    Ok(BurnTransformerBlockW {
        attn: load_qkv(content, reader, &format!("{prefix}.attention"), None)?,
        mlp: load_mlp(content, reader, &format!("{prefix}.mlp"))?,
        ln_w: load_v(content, reader, &format!("{prefix}.layernorm.weight"))?,
        ln_mlp_w: load_v(content, reader, &format!("{prefix}.layernorm_mlp.weight"))?,
    })
}

fn load_icl_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnIclBlockW> {
    let ssmax_prefix = format!("{prefix}.icl_attention.softmax_scaling_layer");
    Ok(BurnIclBlockW {
        attn: load_qkv(
            content,
            reader,
            &format!("{prefix}.icl_attention"),
            Some(&ssmax_prefix),
        )?,
        mlp: load_mlp(content, reader, &format!("{prefix}.mlp"))?,
        ln_w: load_v(content, reader, &format!("{prefix}.layernorm.weight"))?,
        ln_mlp_w: load_v(content, reader, &format!("{prefix}.layernorm_mlp.weight"))?,
    })
}

impl BurnTabPfnModel {
    pub fn load(gguf_path: &Path, config: TabPfnConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let x_embed_w = load_t(&content, &mut reader, "x_embed.weight")?;
        let x_embed_b = load_v(&content, &mut reader, "x_embed.bias")?;
        let col_y_encoder_w = load_t(&content, &mut reader, "col_y_encoder.embedding.weight")?;
        let icl_y_encoder_w = load_t(&content, &mut reader, "icl_y_encoder.embedding.weight")?;

        let mut dist_embed = Vec::with_capacity(config.dist_embed_num_blocks);
        for n in 0..config.dist_embed_num_blocks {
            let p = format!("feature_distribution_embedder.layers.{n}");
            dist_embed.push(BurnIsabW {
                ind_vectors: load_t(&content, &mut reader, &format!("{p}.inducing_vectors"))?,
                block1: load_cross_attn_block(
                    &content,
                    &mut reader,
                    &format!("{p}.cross_attn_block1"),
                    true,
                )?,
                block2: load_cross_attn_block(
                    &content,
                    &mut reader,
                    &format!("{p}.cross_attn_block2"),
                    false,
                )?,
            });
        }

        let mut col_agg_blocks = Vec::with_capacity(config.feat_agg_num_blocks);
        for n in 0..config.feat_agg_num_blocks {
            col_agg_blocks.push(load_transformer_block(
                &content,
                &mut reader,
                &format!("column_aggregator.blocks.{n}"),
            )?);
        }
        let col_agg_cls_tokens = load_t(&content, &mut reader, "column_aggregator.cls_tokens")?;
        let col_agg_out_ln_w = load_v(&content, &mut reader, "column_aggregator.out_ln.weight")?;
        let col_agg_rope_freqs: Vec<f32> = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "column_aggregator.rope.freqs",
            &CDevice::Cpu,
            DType::F32,
        )?
        .flatten_all()?
        .to_vec1()?;

        let mut icl_blocks = Vec::with_capacity(config.nlayers);
        for n in 0..config.nlayers {
            icl_blocks.push(load_icl_block(
                &content,
                &mut reader,
                &format!("icl_blocks.{n}"),
            )?);
        }
        let output_norm_w = load_v(&content, &mut reader, "output_norm.weight")?;
        let many_class_decoder = BurnManyClassDecoderW {
            q_w: load_t(
                &content,
                &mut reader,
                "many_class_decoder.q_projection.weight",
            )?,
            q_b: load_v(
                &content,
                &mut reader,
                "many_class_decoder.q_projection.bias",
            )?,
            k_w: load_t(
                &content,
                &mut reader,
                "many_class_decoder.k_projection.weight",
            )?,
            k_b: load_v(
                &content,
                &mut reader,
                "many_class_decoder.k_projection.bias",
            )?,
            ssmax: if config.decoder_use_softmax_scaling {
                Some(load_ssmax(
                    &content,
                    &mut reader,
                    "many_class_decoder.softmax_scaling_layer",
                )?)
            } else {
                None
            },
        };

        Ok(Self {
            config,
            x_embed_w,
            x_embed_b,
            col_y_encoder_w,
            icl_y_encoder_w,
            dist_embed,
            col_agg_blocks,
            col_agg_cls_tokens,
            col_agg_rope_freqs,
            col_agg_out_ln_w,
            icl_blocks,
            output_norm_w,
            many_class_decoder,
        })
    }

    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let dev = device();
        let train_size = x_support.len();
        let mut all_rows = x_support.to_vec();
        all_rows.extend_from_slice(x_query);
        let t = all_rows.len();
        let h = all_rows[0].len();
        let processed = preprocess_x(&all_rows, train_size);
        let grouped = feature_group_with_nan_indicators(
            &processed,
            h,
            self.config.feature_group_size,
            self.config.use_nan_indicators,
        );
        let cell_dim = grouped[0][0].len();
        let mut flat = vec![0f32; h * t * cell_dim];
        for (row_idx, row) in grouped.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                let base = col_idx * t * cell_dim + row_idx * cell_dim;
                flat[base..base + cell_dim].copy_from_slice(cell);
            }
        }
        let x_grouped = Tensor::<3>::from_data(TensorData::new(flat, [h, t, cell_dim]), &dev);
        let y_col_emb = self.embedding_lookup(y_support, &self.col_y_encoder_w)?;
        let col_out = self.dist_embed_forward(&x_grouped, &y_col_emb, train_size)?;
        let row_out = self.col_agg_forward(&col_out, train_size)?;
        let y_icl_emb = self.embedding_lookup(y_support, &self.icl_y_encoder_w)?;
        let train_part: Tensor<2> = row_out.clone().narrow(0, 0, train_size);
        let train_biased = train_part + y_icl_emb;
        let mut r = if train_size < t {
            Tensor::cat(
                vec![train_biased, row_out.narrow(0, train_size, t - train_size)],
                0,
            )
        } else {
            train_biased
        };
        for blk in &self.icl_blocks {
            r = self.icl_block_forward(&r, blk, train_size)?;
        }
        let r = rms(r, self.output_norm_w.clone());
        let train_emb: Tensor<2> = r.clone().narrow(0, 0, train_size);
        let test_emb: Tensor<2> = r.narrow(0, train_size, t - train_size);
        let highest_target = *y_support
            .iter()
            .max()
            .context("y_support must be non-empty")?;
        let logits = self.many_class_decoder_forward(
            &train_emb,
            &test_emb,
            y_support,
            highest_target,
            n_classes,
        )?;
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

    fn embedding_lookup(&self, y: &[usize], table: &Tensor<2>) -> Result<Tensor<2>> {
        let dim = table.dims()[1];
        let flat: Vec<f32> = to_host(table)?;
        let mut out = vec![0f32; y.len() * dim];
        for (i, &c) in y.iter().enumerate() {
            out[i * dim..(i + 1) * dim].copy_from_slice(&flat[c * dim..(c + 1) * dim]);
        }
        Ok(Tensor::<2>::from_data(
            TensorData::new(out, [y.len(), dim]),
            &device(),
        ))
    }

    fn dist_embed_forward(
        &self,
        x_grouped: &Tensor<3>,
        y_col_emb: &Tensor<2>,
        train_size: usize,
    ) -> Result<Tensor<3>> {
        // x_embed has bias in this checkpoint.
        let cell_emb = linear(
            x_grouped.clone(),
            self.x_embed_w.clone(),
            Some(self.x_embed_b.clone()),
        );
        let t = cell_emb.dims()[1];
        let train_part: Tensor<3> = cell_emb.clone().narrow(1, 0, train_size);
        // y_col_emb [train, D] unsqueezed to [1, train, D] broadcasts over H.
        let yb: Tensor<3> = y_col_emb.clone().unsqueeze_dim(0);
        let train_biased = train_part + yb;
        let mut src = if train_size < t {
            Tensor::cat(
                vec![train_biased, cell_emb.narrow(1, train_size, t - train_size)],
                1,
            )
        } else {
            train_biased
        };
        let n_heads = self.config.dist_embed_num_heads;
        let head_dim = self.config.embed_dim / n_heads;
        for isab in &self.dist_embed {
            let h = src.dims()[0];
            let num_inds = isab.ind_vectors.dims()[0];
            let dim = isab.ind_vectors.dims()[1];
            let ind: Tensor<3> = isab
                .ind_vectors
                .clone()
                .unsqueeze_dim::<3>(0)
                .expand([h, num_inds, dim]);
            let kv_train: Tensor<3> = src.clone().narrow(1, 0, train_size);
            let hidden = cross_attn_block_forward(
                &ind,
                &kv_train,
                &isab.block1,
                n_heads,
                head_dim,
                train_size,
            )?;
            src =
                cross_attn_block_forward(&src, &hidden, &isab.block2, n_heads, head_dim, num_inds)?;
        }
        Ok(src)
    }

    fn col_agg_forward(&self, col_embeddings: &Tensor<3>, _train_size: usize) -> Result<Tensor<2>> {
        let cd = col_embeddings.dims();
        let (h, t) = (cd[0], cd[1]);
        let dim = self.config.embed_dim;
        let n_cls = self.config.feat_agg_num_cls_tokens;
        let n_heads = self.config.feat_agg_num_heads;
        let head_dim = dim / n_heads;
        let feat: Tensor<3> = col_embeddings.clone().permute([1, 0, 2]);
        let cls: Tensor<3> = self
            .col_agg_cls_tokens
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([t, n_cls, dim]);
        let mut seq = Tensor::cat(vec![cls, feat], 1);
        let max_len = h + n_cls;
        let (cos, sin) = rope_table(&self.col_agg_rope_freqs, max_len)?;
        let n_blocks = self.col_agg_blocks.len();
        for (i, blk) in self.col_agg_blocks.iter().enumerate() {
            if i + 1 == n_blocks {
                let cls_q: Tensor<3> = seq.clone().narrow(1, 0, n_cls);
                seq = transformer_block_forward_cross(
                    &cls_q,
                    &seq,
                    blk,
                    n_heads,
                    head_dim,
                    Some((cos.clone(), sin.clone())),
                )?;
            } else {
                seq = transformer_block_forward(
                    &seq,
                    blk,
                    n_heads,
                    head_dim,
                    Some((cos.clone(), sin.clone())),
                )?;
            }
        }
        let out = rms(seq, self.col_agg_out_ln_w.clone());
        Ok(out.reshape([t, n_cls * dim]))
    }

    fn icl_block_forward(
        &self,
        x: &Tensor<2>,
        blk: &BurnIclBlockW,
        train_size: usize,
    ) -> Result<Tensor<2>> {
        let n_heads = self.config.icl_num_heads;
        let icl_dim = self.config.icl_dim();
        let head_dim = icl_dim / n_heads;
        let t = x.dims()[0];
        let normed = rms(x.clone(), blk.ln_w.clone());
        let q = linear_nb(normed.clone(), blk.attn.q_w.clone()).reshape([t, n_heads, head_dim]);
        let x_train: Tensor<2> = normed.narrow(0, 0, train_size);
        let k = linear_nb(x_train.clone(), blk.attn.k_w.clone())
            .reshape([train_size, n_heads, head_dim]);
        let v = linear_nb(x_train, blk.attn.v_w.clone()).reshape([train_size, n_heads, head_dim]);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let v: Tensor<3> = v.permute([1, 0, 2]);
        let attn_out = match self.config.icl_num_kv_heads_test {
            Some(kv_heads_test) if train_size < t => {
                let q_train: Tensor<3> = q.clone().narrow(1, 0, train_size);
                let q_test: Tensor<3> = q.narrow(1, train_size, t - train_size);
                let out_train = sdpa_with_ssmax(
                    &q_train,
                    &k,
                    &v,
                    blk.attn.ssmax.as_ref(),
                    train_size,
                    n_heads,
                    head_dim,
                )?;
                let k_test: Tensor<3> = k.narrow(0, 0, kv_heads_test);
                let v_test: Tensor<3> = v.narrow(0, 0, kv_heads_test);
                let k_test = repeat_heads(&k_test, n_heads / kv_heads_test);
                let v_test = repeat_heads(&v_test, n_heads / kv_heads_test);
                let out_test = sdpa_with_ssmax(
                    &q_test,
                    &k_test,
                    &v_test,
                    blk.attn.ssmax.as_ref(),
                    train_size,
                    n_heads,
                    head_dim,
                )?;
                Tensor::cat(vec![out_train, out_test], 1)
            }
            _ => sdpa_with_ssmax(
                &q,
                &k,
                &v,
                blk.attn.ssmax.as_ref(),
                train_size,
                n_heads,
                head_dim,
            )?,
        };
        let attn_out: Tensor<2> = attn_out.permute([1, 0, 2]).reshape([t, icl_dim]);
        let attn_out = linear_nb(attn_out, blk.attn.out_w.clone());
        let x = x.clone() + attn_out;
        let ff_in = rms(x.clone(), blk.ln_mlp_w.clone());
        let ff = mlp_forward(ff_in, &blk.mlp);
        Ok(x + ff)
    }

    fn many_class_decoder_forward(
        &self,
        train_emb: &Tensor<2>,
        test_emb: &Tensor<2>,
        y_support: &[usize],
        highest_target: usize,
        n_classes: usize,
    ) -> Result<Tensor<2>> {
        let dec = &self.many_class_decoder;
        let n_heads = self.config.decoder_num_heads;
        let head_dim = self.config.decoder_head_dim;
        let n = train_emb.dims()[0];
        let m = test_emb.dims()[0];
        let q = linear(test_emb.clone(), dec.q_w.clone(), Some(dec.q_b.clone()))
            .reshape([m, n_heads, head_dim]);
        let k = linear(train_emb.clone(), dec.k_w.clone(), Some(dec.k_b.clone()))
            .reshape([n, n_heads, head_dim]);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let one_hot_width = highest_target + 1;
        let mut one_hot = vec![0f32; n * one_hot_width];
        for (i, &c) in y_support.iter().enumerate() {
            one_hot[i * one_hot_width + c] = 1.0;
        }
        let dev = device();
        let v_row = Tensor::<2>::from_data(TensorData::new(one_hot, [n, one_hot_width]), &dev);
        let v: Tensor<3> = v_row
            .unsqueeze_dim::<3>(0)
            .expand([n_heads, n, one_hot_width]);
        let num_chunks = one_hot_width.div_ceil(head_dim);
        let padded_width = num_chunks * head_dim;
        let v = if padded_width > one_hot_width {
            let pad = Tensor::<3>::zeros([n_heads, n, padded_width - one_hot_width], &dev);
            Tensor::cat(vec![v, pad], 2)
        } else {
            v
        };
        let mut chunk_outs = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let v_chunk: Tensor<3> = v.clone().narrow(2, c * head_dim, head_dim);
            chunk_outs.push(sdpa_with_ssmax(
                &q,
                &k,
                &v_chunk,
                dec.ssmax.as_ref(),
                n,
                n_heads,
                head_dim,
            )?);
        }
        let out = if chunk_outs.len() == 1 {
            chunk_outs.into_iter().next().unwrap()
        } else {
            let refs: Vec<Tensor<3>> = chunk_outs;
            let mut acc = refs[0].clone();
            for r in refs.iter().skip(1) {
                acc = Tensor::cat(vec![acc, r.clone()], 2);
            }
            acc
        };
        let out: Tensor<3> = out.narrow(2, 0, one_hot_width);
        let out: Tensor<2> = out.mean_dim(0).squeeze_dim(0);
        let out: Tensor<2> = if n_classes > one_hot_width {
            let pad = Tensor::<2>::zeros([m, n_classes - one_hot_width], &dev);
            Tensor::cat(vec![out, pad], 1)
        } else {
            out.narrow(1, 0, n_classes)
        };
        let clamped = out.clamp(1e-5, f32::INFINITY);
        Ok((clamped + 3e-5).log())
    }
}

fn sdpa_with_ssmax(
    q: &Tensor<3>,
    k: &Tensor<3>,
    v: &Tensor<3>,
    ssmax: Option<&BurnSsmaxW>,
    ssmax_n: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor<3>> {
    let q = match ssmax {
        Some(s) => apply_ssmax(q, s, ssmax_n, n_heads, head_dim)?,
        None => q.clone(),
    };
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scores = q.matmul(k.clone().transpose()).mul_scalar(scale);
    let probs = softmax_last(scores);
    Ok(probs.matmul(v.clone()))
}

fn repeat_heads(x: &Tensor<3>, repeat: usize) -> Tensor<3> {
    if repeat == 1 {
        return x.clone();
    }
    let d = x.dims();
    x.clone()
        .unsqueeze_dim::<4>(1)
        .expand([d[0], repeat, d[1], d[2]])
        .reshape([d[0] * repeat, d[1], d[2]])
}

fn apply_ssmax(
    q: &Tensor<3>,
    ssmax: &BurnSsmaxW,
    n: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor<3>> {
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
    let base: Tensor<3> = base.reshape([n_heads, 1, head_dim]);
    let qd = q.dims();
    let (batch_heads, seq) = (qd[0], qd[1]);
    let batch = batch_heads / n_heads;
    let base = if batch > 1 {
        base.unsqueeze_dim::<4>(0)
            .expand([batch, n_heads, 1, head_dim])
            .reshape([batch_heads, 1, head_dim])
    } else {
        base
    };
    let q_flat: Tensor<2> = q.clone().reshape([batch_heads * seq, head_dim]);
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
    let modulation: Tensor<3> = qm
        .tanh()
        .add_scalar(1.0)
        .reshape([batch_heads, seq, head_dim]);
    let scales = base * modulation;
    Ok(q.clone() * scales)
}

fn mlp_forward(x: Tensor<2>, mlp: &BurnMlpW) -> Tensor<2> {
    let h = gelu(linear_nb(x, mlp.fc1_w.clone()));
    linear_nb(h, mlp.fc2_w.clone())
}

fn cross_attn_block_forward(
    query: &Tensor<3>,
    context: &Tensor<3>,
    blk: &BurnCrossAttnBlockW,
    n_heads: usize,
    head_dim: usize,
    ssmax_n: usize,
) -> Result<Tensor<3>> {
    let q_normed = rms(query.clone(), blk.ln_q_w.clone());
    let kv_normed = rms(context.clone(), blk.ln_kv_w.clone());
    let attn_out = batched_qkv_attention(
        &q_normed, &kv_normed, &blk.attn, n_heads, head_dim, ssmax_n, None,
    )?;
    let x = query.clone() + attn_out;
    let ff_in = rms(x.clone(), blk.ln2_w.clone());
    let ff = mlp_forward_2d_to_3d(ff_in, &blk.mlp);
    Ok(x + ff)
}

fn mlp_forward_2d_to_3d(x: Tensor<3>, mlp: &BurnMlpW) -> Tensor<3> {
    let d = x.dims();
    let h = gelu(linear_nb(x.reshape([d[0] * d[1], d[2]]), mlp.fc1_w.clone()));
    linear_nb(h, mlp.fc2_w.clone()).reshape(d)
}

fn transformer_block_forward(
    x: &Tensor<3>,
    blk: &BurnTransformerBlockW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
) -> Result<Tensor<3>> {
    let normed = rms(x.clone(), blk.ln_w.clone());
    let attn_out = batched_qkv_self_attention(&normed, &blk.attn, n_heads, head_dim, rope_cos_sin)?;
    let x = x.clone() + attn_out;
    let ff_in = rms(x.clone(), blk.ln_mlp_w.clone());
    let ff = mlp_forward_2d_to_3d(ff_in, &blk.mlp);
    Ok(x + ff)
}

fn transformer_block_forward_cross(
    query: &Tensor<3>,
    context: &Tensor<3>,
    blk: &BurnTransformerBlockW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
) -> Result<Tensor<3>> {
    let q_normed = rms(query.clone(), blk.ln_w.clone());
    let kv_normed = rms(context.clone(), blk.ln_w.clone());
    let attn_out = batched_qkv_attention(
        &q_normed,
        &kv_normed,
        &blk.attn,
        n_heads,
        head_dim,
        0,
        rope_cos_sin,
    )?;
    let x = query.clone() + attn_out;
    let ff_in = rms(x.clone(), blk.ln_mlp_w.clone());
    let ff = mlp_forward_2d_to_3d(ff_in, &blk.mlp);
    Ok(x + ff)
}

fn batched_qkv_attention(
    q_in: &Tensor<3>,
    kv_in: &Tensor<3>,
    attn: &BurnQkvW,
    n_heads: usize,
    head_dim: usize,
    ssmax_n: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
) -> Result<Tensor<3>> {
    let dim = n_heads * head_dim;
    let qd = q_in.dims();
    let (batch, q_len) = (qd[0], qd[1]);
    let kv_len = kv_in.dims()[1];
    let q = linear_nb(q_in.clone(), attn.q_w.clone()).reshape([batch, q_len, n_heads, head_dim]);
    let k = linear_nb(kv_in.clone(), attn.k_w.clone()).reshape([batch, kv_len, n_heads, head_dim]);
    let v = linear_nb(kv_in.clone(), attn.v_w.clone()).reshape([batch, kv_len, n_heads, head_dim]);
    let q: Tensor<4> = q.permute([0, 2, 1, 3]);
    let k: Tensor<4> = k.permute([0, 2, 1, 3]);
    let v: Tensor<4> = v.permute([0, 2, 1, 3]);
    let (q, k) = match rope_cos_sin {
        Some((cos, sin)) => (
            zsfm_burn::rope::apply_llama_rope(q, cos.clone(), sin.clone(), 0),
            zsfm_burn::rope::apply_llama_rope(k, cos, sin, 0),
        ),
        None => (q, k),
    };
    // Merge batch*heads for the shared 3D sdpa helper.
    let q: Tensor<3> = q
        .permute([0, 2, 1, 3])
        .reshape([batch * n_heads, q_len, head_dim]);
    let k: Tensor<3> = k
        .permute([0, 2, 1, 3])
        .reshape([batch * n_heads, kv_len, head_dim]);
    let v: Tensor<3> = v
        .permute([0, 2, 1, 3])
        .reshape([batch * n_heads, kv_len, head_dim]);
    let n = if ssmax_n > 0 { ssmax_n } else { kv_len };
    let out = sdpa_with_ssmax(&q, &k, &v, attn.ssmax.as_ref(), n, n_heads, head_dim)?;
    let out: Tensor<4> = out
        .reshape([batch, n_heads, q_len, head_dim])
        .permute([0, 2, 1, 3]);
    let out: Tensor<3> = out.reshape([batch, q_len, dim]);
    Ok(linear_nb(out, attn.out_w.clone()))
}

fn batched_qkv_self_attention(
    x: &Tensor<3>,
    attn: &BurnQkvW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(Tensor<2>, Tensor<2>)>,
) -> Result<Tensor<3>> {
    batched_qkv_attention(x, x, attn, n_heads, head_dim, 0, rope_cos_sin)
}

fn rope_table(freqs: &[f32], max_len: usize) -> Result<(Tensor<2>, Tensor<2>)> {
    let dev = device();
    let half = freqs.len();
    let mut cos = vec![0f32; max_len * half * 2];
    let mut sin = vec![0f32; max_len * half * 2];
    for p in 0..max_len {
        for i in 0..half {
            let angle = p as f32 * freqs[i];
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

fn preprocess_x(rows: &[Vec<f32>], train_size: usize) -> Vec<Vec<f32>> {
    let n_feat = rows[0].len();
    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let m: f32 = rows[..train_size].iter().map(|r| r[f]).sum::<f32>() / train_size as f32;
        let denom = if train_size > 1 {
            (train_size - 1) as f32
        } else {
            1.0
        };
        let var: f32 = rows[..train_size]
            .iter()
            .map(|r| (r[f] - m).powi(2))
            .sum::<f32>()
            / denom;
        mean[f] = m;
        std[f] = if var == 0.0 || train_size <= 1 {
            1.0
        } else {
            var.sqrt()
        };
    }
    let eps = f32::EPSILON;
    rows.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(f, &v)| ((v - mean[f]) / (std[f] + eps)).clamp(-100.0, 100.0))
                .collect()
        })
        .collect()
}

fn feature_group_with_nan_indicators(
    rows: &[Vec<f32>],
    h: usize,
    size: usize,
    use_nan_indicators: bool,
) -> Vec<Vec<Vec<f32>>> {
    rows.iter()
        .map(|row| {
            (0..h)
                .map(|g| {
                    let mut cell: Vec<f32> =
                        (0..size).map(|k| row[(g + (1usize << k)) % h]).collect();
                    if use_nan_indicators {
                        cell.extend(std::iter::repeat_n(0.0f32, size));
                    }
                    cell
                })
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

    fn tiny_config() -> TabPfnConfig {
        TabPfnConfig {
            embed_dim: 8,
            dist_embed_num_blocks: 1,
            dist_embed_num_heads: 2,
            dist_embed_num_inducing_points: 2,
            feature_group_size: 2,
            feat_agg_num_blocks: 1,
            feat_agg_num_heads: 2,
            feat_agg_num_cls_tokens: 2,
            nlayers: 1,
            icl_num_heads: 2,
            icl_num_kv_heads_test: Some(1),
            decoder_head_dim: 4,
            decoder_num_heads: 2,
            decoder_use_softmax_scaling: true,
            ff_factor: 2,
            softmax_scaling_mlp_hidden_dim: 8,
            max_num_classes: 2,
            use_nan_indicators: false,
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

    fn put_ssmax(
        w: &mut GGUFWriter,
        prefix: &str,
        base_out: usize,
        hd: usize,
        seed: &mut u64,
        s: f32,
    ) {
        put2(w, &format!("{prefix}.base_mlp.0.weight"), 8, 1, seed, s);
        put1(w, &format!("{prefix}.base_mlp.0.bias"), 8, seed);
        put2(
            w,
            &format!("{prefix}.base_mlp.2.weight"),
            base_out,
            8,
            seed,
            s,
        );
        put1(w, &format!("{prefix}.base_mlp.2.bias"), base_out, seed);
        put2(w, &format!("{prefix}.query_mlp.0.weight"), 8, hd, seed, s);
        put1(w, &format!("{prefix}.query_mlp.0.bias"), 8, seed);
        put2(w, &format!("{prefix}.query_mlp.2.weight"), hd, 8, seed, s);
        put1(w, &format!("{prefix}.query_mlp.2.bias"), hd, seed);
    }

    fn put_qkv(
        w: &mut GGUFWriter,
        prefix: &str,
        d: usize,
        seed: &mut u64,
        s: f32,
        ssmax: Option<(usize, usize)>,
    ) {
        for t in [
            "q_projection",
            "k_projection",
            "v_projection",
            "out_projection",
        ] {
            put2(w, &format!("{prefix}.{t}.weight"), d, d, seed, s);
        }
        if let Some((base_out, hd)) = ssmax {
            put_ssmax(
                w,
                &format!("{prefix}.softmax_scaling_layer"),
                base_out,
                hd,
                seed,
                s,
            );
        }
    }

    fn put_mlp(w: &mut GGUFWriter, prefix: &str, d: usize, seed: &mut u64, s: f32) {
        put2(w, &format!("{prefix}.0.weight"), 2 * d, d, seed, s);
        put2(w, &format!("{prefix}.2.weight"), d, 2 * d, seed, s);
    }

    #[test]
    fn burn_matches_candle_classification() {
        let dir = std::env::temp_dir().join(format!("tabpfn-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tabpfn".into()),
        );
        let mut seed = 2300u64;
        let s = 0.05;
        put2(&mut w, "x_embed.weight", 8, 2, &mut seed, s);
        put1(&mut w, "x_embed.bias", 8, &mut seed);
        put2(&mut w, "col_y_encoder.embedding.weight", 2, 8, &mut seed, s);
        put2(
            &mut w,
            "icl_y_encoder.embedding.weight",
            2,
            16,
            &mut seed,
            s,
        );
        put2(
            &mut w,
            "feature_distribution_embedder.layers.0.inducing_vectors",
            2,
            8,
            &mut seed,
            s,
        );
        let cb1 = "feature_distribution_embedder.layers.0.cross_attn_block1";
        put_qkv(
            &mut w,
            &format!("{cb1}.attn"),
            8,
            &mut seed,
            s,
            Some((8, 4)),
        );
        put_mlp(&mut w, &format!("{cb1}.mlp"), 8, &mut seed, s);
        put1(&mut w, &format!("{cb1}.layernorm_q.weight"), 8, &mut seed);
        put1(&mut w, &format!("{cb1}.layernorm_kv.weight"), 8, &mut seed);
        put1(&mut w, &format!("{cb1}.layernorm2.weight"), 8, &mut seed);
        let cb2 = "feature_distribution_embedder.layers.0.cross_attn_block2";
        put_qkv(&mut w, &format!("{cb2}.attn"), 8, &mut seed, s, None);
        put_mlp(&mut w, &format!("{cb2}.mlp"), 8, &mut seed, s);
        put1(&mut w, &format!("{cb2}.layernorm_q.weight"), 8, &mut seed);
        put1(&mut w, &format!("{cb2}.layernorm_kv.weight"), 8, &mut seed);
        put1(&mut w, &format!("{cb2}.layernorm2.weight"), 8, &mut seed);
        put_qkv(
            &mut w,
            "column_aggregator.blocks.0.attention",
            8,
            &mut seed,
            s,
            None,
        );
        put_mlp(&mut w, "column_aggregator.blocks.0.mlp", 8, &mut seed, s);
        put1(
            &mut w,
            "column_aggregator.blocks.0.layernorm.weight",
            8,
            &mut seed,
        );
        put1(
            &mut w,
            "column_aggregator.blocks.0.layernorm_mlp.weight",
            8,
            &mut seed,
        );
        put2(&mut w, "column_aggregator.cls_tokens", 2, 8, &mut seed, s);
        put1(&mut w, "column_aggregator.out_ln.weight", 8, &mut seed);
        put1(&mut w, "column_aggregator.rope.freqs", 2, &mut seed);
        put_qkv(
            &mut w,
            "icl_blocks.0.icl_attention",
            16,
            &mut seed,
            s,
            Some((16, 8)),
        );
        put_mlp(&mut w, "icl_blocks.0.mlp", 16, &mut seed, s);
        put1(&mut w, "icl_blocks.0.layernorm.weight", 16, &mut seed);
        put1(&mut w, "icl_blocks.0.layernorm_mlp.weight", 16, &mut seed);
        put1(&mut w, "output_norm.weight", 16, &mut seed);
        put2(
            &mut w,
            "many_class_decoder.q_projection.weight",
            8,
            16,
            &mut seed,
            s,
        );
        put1(&mut w, "many_class_decoder.q_projection.bias", 8, &mut seed);
        put2(
            &mut w,
            "many_class_decoder.k_projection.weight",
            8,
            16,
            &mut seed,
            s,
        );
        put1(&mut w, "many_class_decoder.k_projection.bias", 8, &mut seed);
        put_ssmax(
            &mut w,
            "many_class_decoder.softmax_scaling_layer",
            8,
            4,
            &mut seed,
            s,
        );
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::TabPfnModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnTabPfnModel::load(&path, cfg).unwrap();
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
        println!("tabpfn synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "tabpfn Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
