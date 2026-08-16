//! TabPFN-3 inference engine — classification only, single pass (no ensembling). NON-COMMERCIAL
//! WEIGHTS LICENSE (TabPFN-3 Non-Commercial License v1.0): research/internal/benchmarking use
//! only — no production, commercial, or hosted-service use without a separate license from Prior
//! Labs GmbH. See the crate's `Cargo.toml` description.
//!
//! Architecture (three stacked transformers, closely related to TabICL's but with real
//! differences — RMSNorm throughout instead of LayerNorm, unbiased separate Q/K/V projections
//! instead of a packed `in_proj`, no-bias MLPs, and a very different final decoder):
//!
//! 1. **Feature distribution embedder**: each (grouped, NaN-indicator-augmented) feature column
//!    is embedded independently by a shared 3-block Set Transformer (`InducedSelfAttentionBlock`:
//!    learned inducing points cross-attend to the *training* rows only — with a learned
//!    query-aware elementwise attention scale ("SoftmaxScalingMLP", identical in spirit to
//!    TabICL's SSMax) — then the full column cross-attends back to the refined inducing points),
//!    with the training targets folded in beforehand via an orthogonal class-embedding lookup.
//! 2. **Column aggregator**: per row, the (grouped) feature-column embeddings plus 4 learned CLS
//!    tokens attend to each other (non-interleaved RoPE, frequencies stored in the checkpoint);
//!    the final block reads out via CLS-tokens-as-query cross-attention, concatenated into one
//!    `embed_dim * 4` row representation.
//! 3. **ICL transformer**: training targets are folded into their rows' representations (again
//!    via an orthogonal embedding lookup), then a 24-block transformer (SoftmaxScalingMLP on
//!    every block) lets query rows attend to training rows only — with a GQA-style quirk: query
//!    (test) rows attend using only the *first* of the 8 K/V heads (broadcast to all 8 query
//!    heads), while training rows use the full 8 K/V heads. A `ManyClassDecoder` then reads out
//!    a probability-like distribution per class via one more attention pass — Q/K project the
//!    row embeddings, V is the (per-head-broadcast) one-hot training-label encoding, so the
//!    attention output is literally an attention-weighted average of one-hot labels; the result
//!    is log-transformed into logits.
//!
//! Regression (the bar-distribution/quantile head) is out of scope — see the crate's `Cargo.toml`
//! description. NaN/Inf indicator features are always computed (the checkpoint's `x_embed`
//! expects them) but real missing-value handling is not exercised: this port assumes clean
//! (non-NaN) input, matching the scope decision already established for every other model in
//! this workspace.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::config::TabPfnConfig;

/// `nn.RMSNorm`'s default `eps` when unset: the input dtype's machine epsilon. This checkpoint's
/// compute dtype is F32, so `torch.finfo(torch.float32).eps`.
const RMS_EPS: f64 = 1.192_092_9e-7;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct SsmaxW {
    base_fc1_w: Tensor,
    base_fc1_b: Tensor,
    base_fc2_w: Tensor,
    base_fc2_b: Tensor,
    query_fc1_w: Tensor,
    query_fc1_b: Tensor,
    query_fc2_w: Tensor,
    query_fc2_b: Tensor,
}

/// Unbiased separate Q/K/V/out projections (`nn.Linear(..., bias=False)`), shared shape for
/// `Attention`, `CrossAttention`, and `ICLAttention`.
struct QkvW {
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    out_w: Tensor,
    ssmax: Option<SsmaxW>,
}

/// No-bias two-layer GELU MLP (`nn.Sequential(Linear, GELU, Linear)`, both `bias=False`).
struct MlpW {
    fc1_w: Tensor,
    fc2_w: Tensor,
}

/// `CrossAttentionBlock`: pre-norm cross-attention with *separate* Q-side/KV-side RMSNorms
/// (not the same weights reused, unlike TabICL's `MultiheadAttentionBlock`).
struct CrossAttnBlockW {
    attn: QkvW,
    mlp: MlpW,
    ln_q_w: Tensor,
    ln_kv_w: Tensor,
    ln2_w: Tensor,
}

struct IsabW {
    ind_vectors: Tensor,
    block1: CrossAttnBlockW,
    block2: CrossAttnBlockW,
}

/// `TransformerBlock`: pre-norm self-attention (`ColumnAggregator`'s blocks). A single
/// `layernorm` is reused for both query and context sides in the CLS-readout (`forward_cross`)
/// variant, matching the reference's literal `self.layernorm(...)` reuse.
struct TransformerBlockW {
    attn: QkvW,
    mlp: MlpW,
    ln_w: Tensor,
    ln_mlp_w: Tensor,
}

/// `ICLTransformerBlock`: pre-norm `ICLAttention` (train-only K/V, GQA-test quirk) + MLP.
struct IclBlockW {
    attn: QkvW,
    mlp: MlpW,
    ln_w: Tensor,
    ln_mlp_w: Tensor,
}

struct ManyClassDecoderW {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    ssmax: Option<SsmaxW>,
}

pub struct TabPfnModel {
    device: Device,
    config: TabPfnConfig,
    x_embed_w: Tensor,
    x_embed_b: Tensor,
    col_y_encoder_w: Tensor,
    icl_y_encoder_w: Tensor,
    dist_embed: Vec<IsabW>,
    col_agg_blocks: Vec<TransformerBlockW>,
    col_agg_cls_tokens: Tensor,
    col_agg_rope_freqs: Vec<f32>,
    col_agg_out_ln_w: Tensor,
    icl_blocks: Vec<IclBlockW>,
    output_norm_w: Tensor,
    many_class_decoder: ManyClassDecoderW,
}

// ---------------------------------------------------------------------------
// GGUF loading (original, dotted PyTorch tensor names — converted via the *generic*
// `zsfm convert` path, names passed through unchanged; no crate-specific tensor_map/convert).
// ---------------------------------------------------------------------------

fn load_t(content: &gguf_file::Content, reader: &mut (impl Read + Seek), name: &str, device: &Device) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

fn load_ssmax(content: &gguf_file::Content, reader: &mut (impl Read + Seek), prefix: &str, device: &Device) -> Result<SsmaxW> {
    let mut t = |s: &str| load_t(content, reader, &format!("{prefix}.{s}"), device);
    Ok(SsmaxW {
        base_fc1_w: t("base_mlp.0.weight")?,
        base_fc1_b: t("base_mlp.0.bias")?,
        base_fc2_w: t("base_mlp.2.weight")?,
        base_fc2_b: t("base_mlp.2.bias")?,
        query_fc1_w: t("query_mlp.0.weight")?,
        query_fc1_b: t("query_mlp.0.bias")?,
        query_fc2_w: t("query_mlp.2.weight")?,
        query_fc2_b: t("query_mlp.2.bias")?,
    })
}

fn load_qkv(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
    ssmax_prefix: Option<&str>,
) -> Result<QkvW> {
    let mut t = |s: &str| load_t(content, reader, &format!("{prefix}.{s}"), device);
    let q_w = t("q_projection.weight")?;
    let k_w = t("k_projection.weight")?;
    let v_w = t("v_projection.weight")?;
    let out_w = t("out_projection.weight")?;
    let ssmax = match ssmax_prefix {
        Some(p) => Some(load_ssmax(content, reader, p, device)?),
        None => None,
    };
    Ok(QkvW { q_w, k_w, v_w, out_w, ssmax })
}

fn load_mlp(content: &gguf_file::Content, reader: &mut (impl Read + Seek), prefix: &str, device: &Device) -> Result<MlpW> {
    Ok(MlpW {
        fc1_w: load_t(content, reader, &format!("{prefix}.0.weight"), device)?,
        fc2_w: load_t(content, reader, &format!("{prefix}.2.weight"), device)?,
    })
}

fn load_cross_attn_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
    has_ssmax: bool,
) -> Result<CrossAttnBlockW> {
    let ssmax_prefix = format!("{prefix}.attn.softmax_scaling_layer");
    let attn = load_qkv(content, reader, &format!("{prefix}.attn"), device, has_ssmax.then_some(ssmax_prefix.as_str()))?;
    let mlp = load_mlp(content, reader, &format!("{prefix}.mlp"), device)?;
    let ln_q_w = load_t(content, reader, &format!("{prefix}.layernorm_q.weight"), device)?;
    let ln_kv_w = load_t(content, reader, &format!("{prefix}.layernorm_kv.weight"), device)?;
    let ln2_w = load_t(content, reader, &format!("{prefix}.layernorm2.weight"), device)?;
    Ok(CrossAttnBlockW { attn, mlp, ln_q_w, ln_kv_w, ln2_w })
}

fn load_transformer_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<TransformerBlockW> {
    let attn = load_qkv(content, reader, &format!("{prefix}.attention"), device, None)?;
    let mlp = load_mlp(content, reader, &format!("{prefix}.mlp"), device)?;
    let ln_w = load_t(content, reader, &format!("{prefix}.layernorm.weight"), device)?;
    let ln_mlp_w = load_t(content, reader, &format!("{prefix}.layernorm_mlp.weight"), device)?;
    Ok(TransformerBlockW { attn, mlp, ln_w, ln_mlp_w })
}

fn load_icl_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
) -> Result<IclBlockW> {
    let ssmax_prefix = format!("{prefix}.icl_attention.softmax_scaling_layer");
    let attn = load_qkv(content, reader, &format!("{prefix}.icl_attention"), device, Some(&ssmax_prefix))?;
    let mlp = load_mlp(content, reader, &format!("{prefix}.mlp"), device)?;
    let ln_w = load_t(content, reader, &format!("{prefix}.layernorm.weight"), device)?;
    let ln_mlp_w = load_t(content, reader, &format!("{prefix}.layernorm_mlp.weight"), device)?;
    Ok(IclBlockW { attn, mlp, ln_w, ln_mlp_w })
}

impl TabPfnModel {
    pub fn load(gguf_path: &Path, config: TabPfnConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let x_embed_w = load_t(&content, &mut reader, "x_embed.weight", &device)?;
        let x_embed_b = load_t(&content, &mut reader, "x_embed.bias", &device)?;
        let col_y_encoder_w = load_t(&content, &mut reader, "col_y_encoder.embedding.weight", &device)?;
        let icl_y_encoder_w = load_t(&content, &mut reader, "icl_y_encoder.embedding.weight", &device)?;

        let mut dist_embed = Vec::with_capacity(config.dist_embed_num_blocks);
        for n in 0..config.dist_embed_num_blocks {
            let p = format!("feature_distribution_embedder.layers.{n}");
            let ind_vectors = load_t(&content, &mut reader, &format!("{p}.inducing_vectors"), &device)?;
            let block1 = load_cross_attn_block(&content, &mut reader, &format!("{p}.cross_attn_block1"), &device, true)?;
            let block2 = load_cross_attn_block(&content, &mut reader, &format!("{p}.cross_attn_block2"), &device, false)?;
            dist_embed.push(IsabW { ind_vectors, block1, block2 });
        }

        let mut col_agg_blocks = Vec::with_capacity(config.feat_agg_num_blocks);
        for n in 0..config.feat_agg_num_blocks {
            let p = format!("column_aggregator.blocks.{n}");
            col_agg_blocks.push(load_transformer_block(&content, &mut reader, &p, &device)?);
        }
        let col_agg_cls_tokens = load_t(&content, &mut reader, "column_aggregator.cls_tokens", &device)?;
        let col_agg_out_ln_w = load_t(&content, &mut reader, "column_aggregator.out_ln.weight", &device)?;
        let rope_freqs_t = load_t(&content, &mut reader, "column_aggregator.rope.freqs", &device)?;
        let col_agg_rope_freqs: Vec<f32> = rope_freqs_t.flatten_all()?.to_vec1()?;

        let mut icl_blocks = Vec::with_capacity(config.nlayers);
        for n in 0..config.nlayers {
            let p = format!("icl_blocks.{n}");
            icl_blocks.push(load_icl_block(&content, &mut reader, &p, &device)?);
        }

        let output_norm_w = load_t(&content, &mut reader, "output_norm.weight", &device)?;

        let decoder_ssmax_prefix = "many_class_decoder.softmax_scaling_layer";
        let many_class_decoder = ManyClassDecoderW {
            q_w: load_t(&content, &mut reader, "many_class_decoder.q_projection.weight", &device)?,
            q_b: load_t(&content, &mut reader, "many_class_decoder.q_projection.bias", &device)?,
            k_w: load_t(&content, &mut reader, "many_class_decoder.k_projection.weight", &device)?,
            k_b: load_t(&content, &mut reader, "many_class_decoder.k_projection.bias", &device)?,
            ssmax: if config.decoder_use_softmax_scaling {
                Some(load_ssmax(&content, &mut reader, decoder_ssmax_prefix, &device)?)
            } else {
                None
            },
        };

        Ok(Self {
            device,
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

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Zero-shot classification. `y_support` are class indices (must satisfy
    /// `max(y_support) < max_num_classes`). Returns probabilities `[n_query][n_classes]`.
    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let train_size = x_support.len();
        let mut all_rows = x_support.to_vec();
        all_rows.extend_from_slice(x_query);
        let t = all_rows.len();
        let h = all_rows[0].len();

        let processed = preprocess_x(&all_rows, train_size);
        let group_size = self.config.feature_group_size;
        let grouped = feature_group_with_nan_indicators(&processed, h, group_size, self.config.use_nan_indicators);
        let cell_dim = grouped[0][0].len();

        let mut flat = vec![0f32; h * t * cell_dim];
        for (row_idx, row) in grouped.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                let base = col_idx * t * cell_dim + row_idx * cell_dim;
                flat[base..base + cell_dim].copy_from_slice(cell);
            }
        }
        let x_grouped = Tensor::from_vec(flat, (h, t, cell_dim), &self.device)?;

        let y_col_emb = self.embedding_lookup(y_support, &self.col_y_encoder_w)?; // (train_size, embed_dim)
        let col_out = self.dist_embed_forward(&x_grouped, &y_col_emb, train_size)?; // (H, T, embed_dim)
        let row_out = self.col_agg_forward(&col_out, train_size)?; // (T, icl_dim)

        let y_icl_emb = self.embedding_lookup(y_support, &self.icl_y_encoder_w)?; // (train_size, icl_dim)
        let train_part = row_out.narrow(0, 0, train_size)?.broadcast_add(&y_icl_emb)?;
        let mut r = if train_size < t {
            Tensor::cat(&[&train_part, &row_out.narrow(0, train_size, t - train_size)?], 0)?
        } else {
            train_part
        };
        for blk in &self.icl_blocks {
            r = self.icl_block_forward(&r, blk, train_size)?;
        }
        let r = zsfm_nn::rms_norm(&r, Some(&self.output_norm_w), RMS_EPS)?;

        let train_emb = r.narrow(0, 0, train_size)?;
        let test_emb = r.narrow(0, train_size, t - train_size)?;

        let highest_target = *y_support.iter().max().context("y_support must be non-empty")?;
        let logits = self.many_class_decoder_forward(&train_emb, &test_emb, y_support, highest_target, n_classes)?;

        let flat_logits: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let n_query = t - train_size;
        const TEMPERATURE: f32 = 0.9;
        let mut result = Vec::with_capacity(n_query);
        for i in 0..n_query {
            let row = &flat_logits[i * n_classes..(i + 1) * n_classes];
            let scaled: Vec<f32> = row.iter().map(|&v| v / TEMPERATURE).collect();
            result.push(softmax(&scaled));
        }
        Ok(result)
    }

    fn embedding_lookup(&self, y: &[usize], table: &Tensor) -> Result<Tensor> {
        let dim = table.dim(1)?;
        let idx = Tensor::from_vec(y.iter().map(|&c| c as u32).collect::<Vec<_>>(), y.len(), &self.device)?;
        table.index_select(&idx, 0)?.reshape((y.len(), dim)).map_err(anyhow::Error::from)
    }

    /// `x_grouped`: `[H, T, cell_dim]`. `y_col_emb`: `[train_size, embed_dim]`. Returns `[H, T,
    /// embed_dim]`.
    fn dist_embed_forward(&self, x_grouped: &Tensor, y_col_emb: &Tensor, train_size: usize) -> Result<Tensor> {
        let cell_emb = zsfm_nn::linear_bias(x_grouped, &self.x_embed_w, &self.x_embed_b)?;
        let t = cell_emb.dim(1)?;
        let train_part = cell_emb.narrow(1, 0, train_size)?.broadcast_add(&y_col_emb.unsqueeze(0)?)?;
        let mut src = if train_size < t {
            Tensor::cat(&[&train_part, &cell_emb.narrow(1, train_size, t - train_size)?], 1)?
        } else {
            train_part
        };

        let n_heads = self.config.dist_embed_num_heads;
        let head_dim = self.config.embed_dim / n_heads;
        for isab in &self.dist_embed {
            let h = src.dim(0)?;
            let num_inds = isab.ind_vectors.dim(0)?;
            let dim = isab.ind_vectors.dim(1)?;
            let ind = isab.ind_vectors.unsqueeze(0)?.expand((h, num_inds, dim))?.contiguous()?;
            let kv_train = src.narrow(1, 0, train_size)?;
            let hidden = cross_attn_block_forward(&ind, &kv_train, &isab.block1, n_heads, head_dim, train_size)?;
            src = cross_attn_block_forward(&src, &hidden, &isab.block2, n_heads, head_dim, num_inds)?;
        }
        Ok(src)
    }

    /// `col_embeddings`: `[H, T, embed_dim]`. Returns row representations `[T, icl_dim]`.
    fn col_agg_forward(&self, col_embeddings: &Tensor, _train_size: usize) -> Result<Tensor> {
        let h = col_embeddings.dim(0)?;
        let t = col_embeddings.dim(1)?;
        let dim = self.config.embed_dim;
        let n_cls = self.config.feat_agg_num_cls_tokens;
        let n_heads = self.config.feat_agg_num_heads;
        let head_dim = dim / n_heads;

        let feat = col_embeddings.permute((1, 0, 2))?.contiguous()?; // (T, H, dim)
        let cls = self.col_agg_cls_tokens.unsqueeze(0)?.expand((t, n_cls, dim))?.contiguous()?;
        let mut seq = Tensor::cat(&[&cls, &feat], 1)?; // (T, H+C, dim)

        let max_len = h + n_cls;
        let (cos, sin) = rope_table(&self.col_agg_rope_freqs, max_len, &self.device)?;

        let n_blocks = self.col_agg_blocks.len();
        for (i, blk) in self.col_agg_blocks.iter().enumerate() {
            if i + 1 == n_blocks {
                let cls_q = seq.narrow(1, 0, n_cls)?;
                seq = transformer_block_forward_cross(&cls_q, &seq, blk, n_heads, head_dim, Some((&cos, &sin)))?;
            } else {
                seq = transformer_block_forward(&seq, blk, n_heads, head_dim, Some((&cos, &sin)))?;
            }
        }

        let out = zsfm_nn::rms_norm(&seq, Some(&self.col_agg_out_ln_w), RMS_EPS)?; // (T, C, dim)
        out.reshape((t, n_cls * dim)).map_err(anyhow::Error::from)
    }

    /// One `ICLTransformerBlock`: train rows use full-head K/V, test rows use only the first
    /// `icl_num_kv_heads_test` head(s) of K/V (broadcast to all query heads) — a GQA-style
    /// reduction that applies to test rows *only*. `x`: `[T, icl_dim]`.
    fn icl_block_forward(&self, x: &Tensor, blk: &IclBlockW, train_size: usize) -> Result<Tensor> {
        let n_heads = self.config.icl_num_heads;
        let icl_dim = self.config.icl_dim();
        let head_dim = icl_dim / n_heads;
        let t = x.dim(0)?;

        let normed = zsfm_nn::rms_norm(x, Some(&blk.ln_w), RMS_EPS)?;
        let q = zsfm_nn::linear_nobias(&normed, &blk.attn.q_w)?.reshape((t, n_heads, head_dim))?;
        let x_train = normed.narrow(0, 0, train_size)?;
        let k = zsfm_nn::linear_nobias(&x_train, &blk.attn.k_w)?.reshape((train_size, n_heads, head_dim))?;
        let v = zsfm_nn::linear_nobias(&x_train, &blk.attn.v_w)?.reshape((train_size, n_heads, head_dim))?;

        // (seq, heads, hd) -> (heads, seq, hd)
        let q = q.permute((1, 0, 2))?.contiguous()?;
        let k = k.permute((1, 0, 2))?.contiguous()?;
        let v = v.permute((1, 0, 2))?.contiguous()?;

        let attn_out = match self.config.icl_num_kv_heads_test {
            Some(kv_heads_test) if train_size < t => {
                let q_train = q.narrow(1, 0, train_size)?;
                let q_test = q.narrow(1, train_size, t - train_size)?;
                let out_train = sdpa_with_ssmax(&q_train, &k, &v, blk.attn.ssmax.as_ref(), train_size, n_heads, head_dim)?;

                let k_test = k.narrow(0, 0, kv_heads_test)?;
                let v_test = v.narrow(0, 0, kv_heads_test)?;
                let k_test = repeat_heads(&k_test, n_heads / kv_heads_test)?;
                let v_test = repeat_heads(&v_test, n_heads / kv_heads_test)?;
                let out_test =
                    sdpa_with_ssmax(&q_test, &k_test, &v_test, blk.attn.ssmax.as_ref(), train_size, n_heads, head_dim)?;
                Tensor::cat(&[&out_train, &out_test], 1)? // (heads, T, hd)
            }
            _ => sdpa_with_ssmax(&q, &k, &v, blk.attn.ssmax.as_ref(), train_size, n_heads, head_dim)?,
        };

        let attn_out = attn_out.permute((1, 0, 2))?.contiguous()?.reshape((t, icl_dim))?;
        let attn_out = zsfm_nn::linear_nobias(&attn_out, &blk.attn.out_w)?;
        let x = (x + attn_out)?;

        let ff_in = zsfm_nn::rms_norm(&x, Some(&blk.ln_mlp_w), RMS_EPS)?;
        let ff = mlp_forward(&ff_in, &blk.mlp)?;
        Ok((x + ff)?)
    }

    /// `train_emb`/`test_emb`: `[N, icl_dim]`/`[M, icl_dim]`. Returns logits `[M, n_classes]`.
    fn many_class_decoder_forward(
        &self,
        train_emb: &Tensor,
        test_emb: &Tensor,
        y_support: &[usize],
        highest_target: usize,
        n_classes: usize,
    ) -> Result<Tensor> {
        let dec = &self.many_class_decoder;
        let n_heads = self.config.decoder_num_heads;
        let head_dim = self.config.decoder_head_dim;
        let n = train_emb.dim(0)?;
        let m = test_emb.dim(0)?;

        let q = zsfm_nn::linear_bias(test_emb, &dec.q_w, &dec.q_b)?.reshape((m, n_heads, head_dim))?;
        let k = zsfm_nn::linear_bias(train_emb, &dec.k_w, &dec.k_b)?.reshape((n, n_heads, head_dim))?;
        let q = q.permute((1, 0, 2))?.contiguous()?; // (heads, M, hd)
        let k = k.permute((1, 0, 2))?.contiguous()?; // (heads, N, hd)

        let one_hot_width = highest_target + 1;
        let mut one_hot = vec![0f32; n * one_hot_width];
        for (i, &c) in y_support.iter().enumerate() {
            one_hot[i * one_hot_width + c] = 1.0;
        }
        // V is shared identically across all heads (no v_projection in ManyClassDecoder).
        let v_row = Tensor::from_vec(one_hot, (n, one_hot_width), &self.device)?;
        let v = v_row.unsqueeze(0)?.expand((n_heads, n, one_hot_width))?.contiguous()?;

        let num_chunks = one_hot_width.div_ceil(head_dim);
        let padded_width = num_chunks * head_dim;
        let v = if padded_width > one_hot_width {
            v.pad_with_zeros(2, 0, padded_width - one_hot_width)?
        } else {
            v
        };

        let mut chunk_outs = Vec::with_capacity(num_chunks);
        for c in 0..num_chunks {
            let v_chunk = v.narrow(2, c * head_dim, head_dim)?;
            let out = sdpa_with_ssmax(&q, &k, &v_chunk, dec.ssmax.as_ref(), n, n_heads, head_dim)?; // (heads, M, hd)
            chunk_outs.push(out);
        }
        let out = if chunk_outs.len() == 1 {
            chunk_outs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Tensor> = chunk_outs.iter().collect();
            Tensor::cat(&refs, 2)? // (heads, M, num_chunks*hd)
        };
        let out = out.narrow(2, 0, one_hot_width)?; // (heads, M, one_hot_width)
        let out = out.mean(0)?; // average over heads -> (M, one_hot_width)

        let out = if n_classes > one_hot_width {
            out.pad_with_zeros(1, 0, n_classes - one_hot_width)?
        } else {
            out.narrow(1, 0, n_classes)?
        };

        let clamped = out.clamp(1e-5f32, f32::INFINITY)?;
        ((clamped + 3e-5)?).log().map_err(anyhow::Error::from)
    }
}

/// Scaled dot-product attention with optional query-aware SSMax scaling. `q`/`k`/`v`:
/// `[heads, seq, hd]`. `ssmax_n` is the KV sequence length used for SSMax's `log(n)` term.
fn sdpa_with_ssmax(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    ssmax: Option<&SsmaxW>,
    ssmax_n: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let q = match ssmax {
        Some(s) => apply_ssmax(q, s, ssmax_n, n_heads, head_dim)?,
        None => q.clone(),
    };
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(v).map_err(anyhow::Error::from)
}

/// `x`: `[kv_heads, seq, hd]` -> `[kv_heads * repeat, seq, hd]` (`repeat_interleave` along the
/// head axis, matching torch's GQA broadcast on the math-backend path).
fn repeat_heads(x: &Tensor, repeat: usize) -> Result<Tensor> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let (h, s, d) = x.dims3()?;
    x.unsqueeze(1)?.expand((h, repeat, s, d))?.reshape((h * repeat, s, d)).map_err(anyhow::Error::from)
}

/// Query-aware elementwise softmax scaling (`SoftmaxScalingMLP`): `q * base_mlp(log n) * (1 +
/// tanh(query_mlp(q)))`. `q`: `[heads, seq, hd]`.
fn apply_ssmax(q: &Tensor, ssmax: &SsmaxW, n: usize, n_heads: usize, head_dim: usize) -> Result<Tensor> {
    let device = q.device();
    let logn = (n.max(1) as f32).ln();
    let logn_t = Tensor::from_vec(vec![logn], (1, 1), device)?;
    let base = zsfm_nn::linear_bias(&logn_t, &ssmax.base_fc1_w, &ssmax.base_fc1_b)?.gelu_erf()?;
    let base = zsfm_nn::linear_bias(&base, &ssmax.base_fc2_w, &ssmax.base_fc2_b)?; // (1, n_heads*head_dim)
    let base = base.reshape((n_heads, 1, head_dim))?;

    // `q`'s leading dim is `batch * n_heads` flattened (batch = 1 for the ICL/decoder stages,
    // batch = H columns for the distribution embedder's per-column ISAB) — `base` only varies
    // per head, so it must be tiled across `batch` before broadcasting against `q`.
    let batch_heads = q.dim(0)?;
    let seq = q.dim(1)?;
    let batch = batch_heads / n_heads;
    let base = if batch > 1 {
        base.unsqueeze(0)?.expand((batch, n_heads, 1, head_dim))?.reshape((batch_heads, 1, head_dim))?
    } else {
        base
    };

    let q_flat = q.reshape((batch_heads * seq, head_dim))?;
    let qm = zsfm_nn::linear_bias(&q_flat, &ssmax.query_fc1_w, &ssmax.query_fc1_b)?.gelu_erf()?;
    let qm = zsfm_nn::linear_bias(&qm, &ssmax.query_fc2_w, &ssmax.query_fc2_b)?;
    let modulation = (qm.tanh()? + 1.0)?.reshape((batch_heads, seq, head_dim))?;

    let scales = base.broadcast_mul(&modulation)?;
    Ok(q.broadcast_mul(&scales)?)
}

fn mlp_forward(x: &Tensor, mlp: &MlpW) -> Result<Tensor> {
    let h = zsfm_nn::linear_nobias(x, &mlp.fc1_w)?.gelu_erf()?;
    zsfm_nn::linear_nobias(&h, &mlp.fc2_w).map_err(anyhow::Error::from)
}

/// `CrossAttentionBlock`: `x = query + Attn(LN_q(query), LN_kv(context))`, then `x = x +
/// MLP(LN2(x))`. `query`/`context`: `[batch, seq, dim]`, `batch` is the induced-self-attention
/// block's "H columns as batch" axis.
fn cross_attn_block_forward(
    query: &Tensor,
    context: &Tensor,
    blk: &CrossAttnBlockW,
    n_heads: usize,
    head_dim: usize,
    ssmax_n: usize,
) -> Result<Tensor> {
    let q_normed = zsfm_nn::rms_norm(query, Some(&blk.ln_q_w), RMS_EPS)?;
    let kv_normed = zsfm_nn::rms_norm(context, Some(&blk.ln_kv_w), RMS_EPS)?;
    let attn_out = batched_qkv_attention(&q_normed, &kv_normed, &blk.attn, n_heads, head_dim, ssmax_n, None)?;
    let x = (query + attn_out)?;
    let ff_in = zsfm_nn::rms_norm(&x, Some(&blk.ln2_w), RMS_EPS)?;
    let ff = mlp_forward(&ff_in, &blk.mlp)?;
    Ok((x + ff)?)
}

/// `TransformerBlock` self-attention: `x = x + Attn(LN(x))`, then `x = x + MLP(LN_mlp(x))`.
/// `x`: `[batch, seq, dim]`, `batch` is the "T rows as batch" axis (`ColumnAggregator`).
fn transformer_block_forward(
    x: &Tensor,
    blk: &TransformerBlockW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
) -> Result<Tensor> {
    let normed = zsfm_nn::rms_norm(x, Some(&blk.ln_w), RMS_EPS)?;
    let attn_out = batched_qkv_self_attention(&normed, &blk.attn, n_heads, head_dim, rope_cos_sin)?;
    let x = (x + attn_out)?;
    let ff_in = zsfm_nn::rms_norm(&x, Some(&blk.ln_mlp_w), RMS_EPS)?;
    let ff = mlp_forward(&ff_in, &blk.mlp)?;
    Ok((x + ff)?)
}

/// `TransformerBlock.forward_cross`: CLS-tokens-as-query readout. The *same* `layernorm` weight
/// normalizes both the query and the context (matching the reference's literal
/// `self.layernorm(...)` reuse on both sides — not a separate `layernorm_kv`).
fn transformer_block_forward_cross(
    query: &Tensor,
    context: &Tensor,
    blk: &TransformerBlockW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
) -> Result<Tensor> {
    let q_normed = zsfm_nn::rms_norm(query, Some(&blk.ln_w), RMS_EPS)?;
    let kv_normed = zsfm_nn::rms_norm(context, Some(&blk.ln_w), RMS_EPS)?;
    let attn_out = batched_qkv_attention(&q_normed, &kv_normed, &blk.attn, n_heads, head_dim, 0, rope_cos_sin)?;
    let x = (query + attn_out)?;
    let ff_in = zsfm_nn::rms_norm(&x, Some(&blk.ln_mlp_w), RMS_EPS)?;
    let ff = mlp_forward(&ff_in, &blk.mlp)?;
    Ok((x + ff)?)
}

/// Cross-attention over an explicit `[batch, seq, dim]` batch axis (the induced-self-attention
/// "H columns as batch" axis, or `ColumnAggregator`'s "T rows as batch" axis for the CLS
/// readout, where RoPE applies to both Q and K — `forward_cross` rotates both sides using each
/// tensor's own sequence length, so the CLS query and the full K sequence get their true
/// absolute positions).
fn batched_qkv_attention(
    q_in: &Tensor,
    kv_in: &Tensor,
    attn: &QkvW,
    n_heads: usize,
    head_dim: usize,
    ssmax_n: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
) -> Result<Tensor> {
    let dim = n_heads * head_dim;
    let batch = q_in.dim(0)?;
    let q_len = q_in.dim(1)?;
    let kv_len = kv_in.dim(1)?;

    let q = zsfm_nn::linear_nobias(q_in, &attn.q_w)?.reshape((batch, q_len, n_heads, head_dim))?;
    let k = zsfm_nn::linear_nobias(kv_in, &attn.k_w)?.reshape((batch, kv_len, n_heads, head_dim))?;
    let v = zsfm_nn::linear_nobias(kv_in, &attn.v_w)?.reshape((batch, kv_len, n_heads, head_dim))?;

    let q = q.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, q_len, head_dim))?;
    let k = k.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, kv_len, head_dim))?;
    let v = v.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, kv_len, head_dim))?;

    let (q, k) = if let Some((cos, sin)) = rope_cos_sin {
        (apply_rope(&q, cos, sin, batch, n_heads)?, apply_rope(&k, cos, sin, batch, n_heads)?)
    } else {
        (q, k)
    };

    let n = if ssmax_n > 0 { ssmax_n } else { kv_len };
    let out = sdpa_with_ssmax(&q, &k, &v, attn.ssmax.as_ref(), n, n_heads, head_dim)?;
    let out = out.reshape((batch, n_heads, q_len, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?.reshape((
        batch,
        q_len,
        dim,
    ))?;
    zsfm_nn::linear_nobias(&out, &attn.out_w).map_err(anyhow::Error::from)
}

/// Self-attention with optional RoPE over an explicit `[batch, seq, dim]` batch axis.
fn batched_qkv_self_attention(
    x: &Tensor,
    attn: &QkvW,
    n_heads: usize,
    head_dim: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
) -> Result<Tensor> {
    let dim = n_heads * head_dim;
    let batch = x.dim(0)?;
    let seq = x.dim(1)?;

    let q = zsfm_nn::linear_nobias(x, &attn.q_w)?.reshape((batch, seq, n_heads, head_dim))?;
    let k = zsfm_nn::linear_nobias(x, &attn.k_w)?.reshape((batch, seq, n_heads, head_dim))?;
    let v = zsfm_nn::linear_nobias(x, &attn.v_w)?.reshape((batch, seq, n_heads, head_dim))?;

    let q = q.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, seq, head_dim))?;
    let k = k.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, seq, head_dim))?;
    let v = v.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch * n_heads, seq, head_dim))?;

    let (q, k) = if let Some((cos, sin)) = rope_cos_sin {
        (apply_rope(&q, cos, sin, batch, n_heads)?, apply_rope(&k, cos, sin, batch, n_heads)?)
    } else {
        (q, k)
    };

    let out = sdpa_with_ssmax(&q, &k, &v, attn.ssmax.as_ref(), seq, n_heads, head_dim)?;
    let out =
        out.reshape((batch, n_heads, seq, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch, seq, dim))?;
    zsfm_nn::linear_nobias(&out, &attn.out_w).map_err(anyhow::Error::from)
}

/// Non-interleaved RoPE. `x`: `[batch*heads, seq, head_dim]`. `cos`/`sin`: `[max_len, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, _batch: usize, _n_heads: usize) -> Result<Tensor> {
    let seq = x.dim(1)?;
    let head_dim = x.dim(2)?;
    let half = head_dim / 2;
    let cos = cos.narrow(0, 0, seq)?.unsqueeze(0)?; // (1, seq, hd)
    let sin = sin.narrow(0, 0, seq)?.unsqueeze(0)?;
    let x1 = x.narrow(2, 0, half)?;
    let x2 = x.narrow(2, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 2)?;
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

fn rope_table(freqs: &[f32], max_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
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
    let cos_t = Tensor::from_vec(cos, (max_len, 2 * half), device)?;
    let sin_t = Tensor::from_vec(sin, (max_len, 2 * half), device)?;
    Ok((cos_t, sin_t))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Mean-impute (no-op here — this port assumes no NaNs, matching the scope decision established
/// for every other model in this workspace) + standardize with `ddof=1` (sample std, matching
/// `torch_nanstd`'s `N-1` correction — *not* the `ddof=0` used by Mitra/TabDPT/TabICL's simpler
/// preprocessing), `+ eps` before dividing, then clip to `[-100, 100]`.
fn preprocess_x(rows: &[Vec<f32>], train_size: usize) -> Vec<Vec<f32>> {
    let n_feat = rows[0].len();
    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let m: f32 = rows[..train_size].iter().map(|r| r[f]).sum::<f32>() / train_size as f32;
        let denom = if train_size > 1 { (train_size - 1) as f32 } else { 1.0 };
        let var: f32 = rows[..train_size].iter().map(|r| (r[f] - m).powi(2)).sum::<f32>() / denom;
        mean[f] = m;
        std[f] = if var == 0.0 || train_size <= 1 { 1.0 } else { var.sqrt() };
    }
    let eps = f32::EPSILON;
    rows.iter()
        .map(|row| {
            row.iter().enumerate().map(|(f, &v)| ((v - mean[f]) / (std[f] + eps)).clamp(-100.0, 100.0)).collect()
        })
        .collect()
}

/// Circular-permutation feature grouping (same formula as TabICL's `feature_group_same`: group
/// `g`'s values are input features at offsets `2^0, 2^1, .., 2^(size-1)` past `g`, mod `H`),
/// with NaN/Inf indicator features (always `0.0` in this no-NaN-support port) concatenated after
/// the real grouped values, matching `x_grouped = cat([x_grouped, ind_grouped], dim=-1)`.
fn feature_group_with_nan_indicators(rows: &[Vec<f32>], h: usize, size: usize, use_nan_indicators: bool) -> Vec<Vec<Vec<f32>>> {
    rows.iter()
        .map(|row| {
            (0..h)
                .map(|g| {
                    let mut cell: Vec<f32> = (0..size).map(|k| row[(g + (1usize << k)) % h]).collect();
                    if use_nan_indicators {
                        cell.extend(std::iter::repeat_n(0.0f32, size));
                    }
                    cell
                })
                .collect()
        })
        .collect()
}
