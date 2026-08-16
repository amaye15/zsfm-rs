//! TabICL v2 inference engine — classification only, single pass (no ensembling), no >10-class
//! mixed-radix/hierarchical path. Architecture: three stacked transformers —
//!
//! 1. **Column embedding**: each (grouped) feature column is embedded independently by a
//!    shared 3-block Set Transformer (`InducedSelfAttentionBlock`: learned inducing points
//!    cross-attend to the *training* rows only, then the full column cross-attends back to
//!    the refined inducing points — `O(n)` instead of `O(n²)`), with the training targets
//!    folded in beforehand ("target-aware") and a learned query-aware elementwise attention
//!    scale ("SSMax") on the first attention stage only.
//! 2. **Row interaction**: per row, the H (grouped) feature-column embeddings plus 4 learned
//!    CLS tokens attend to each other (non-interleaved RoPE over the H+4 position index); the
//!    final block reads out via CLS-tokens-as-query cross-attention, concatenated into one
//!    `embed_dim * 4` row representation.
//! 2. **In-context learning**: training targets are folded into their rows' representations,
//!    then a 12-block transformer (SSMax on every block) lets query rows attend to training
//!    rows only, followed by a 2-layer decoder head.
//!
//! Regression (`quantile_dist.py`'s monotonic quantile-distribution head) and the >10-class
//! mixed-radix/hierarchical classification path are out of scope — see the crate's `Cargo.toml`
//! description.

use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::config::TabIclConfig;

const LN_EPS: f64 = 1e-5;

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

struct AttnW {
    in_proj_w: Tensor,
    in_proj_b: Tensor,
    out_proj_w: Tensor,
    out_proj_b: Tensor,
    ssmax: Option<SsmaxW>,
}

struct BlockW {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    attn: AttnW,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

struct IsabW {
    ind_vectors: Tensor,
    attn1: BlockW,
    attn2: BlockW,
}

struct ColEmbedderW {
    in_linear_w: Tensor,
    in_linear_b: Tensor,
    y_encoder_w: Tensor,
    y_encoder_b: Tensor,
    blocks: Vec<IsabW>,
}

struct RowInteractorW {
    cls_tokens: Tensor,
    blocks: Vec<BlockW>,
    out_ln_w: Tensor,
    out_ln_b: Tensor,
    rope_freqs: Vec<f32>,
}

struct IclPredictorW {
    y_encoder_w: Tensor,
    y_encoder_b: Tensor,
    blocks: Vec<BlockW>,
    ln_w: Tensor,
    ln_b: Tensor,
    decoder_fc1_w: Tensor,
    decoder_fc1_b: Tensor,
    decoder_fc2_w: Tensor,
    decoder_fc2_b: Tensor,
}

pub struct TabIclModel {
    device: Device,
    config: TabIclConfig,
    col: ColEmbedderW,
    row: RowInteractorW,
    icl: IclPredictorW,
}

// ---------------------------------------------------------------------------
// GGUF loading (original, dotted PyTorch tensor names — this checkpoint was converted via the
// *generic* `zsfm convert` path, which passes names through unchanged, so there's no
// crate-specific tensor_map/convert step).
// ---------------------------------------------------------------------------

fn load_t(content: &gguf_file::Content, reader: &mut (impl Read + Seek), name: &str, device: &Device) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
    has_ssmax: bool,
) -> Result<AttnW> {
    let mut t = |s: &str| load_t(content, reader, &format!("{prefix}.{s}"), device);
    let in_proj_w = t("in_proj_weight")?;
    let in_proj_b = t("in_proj_bias")?;
    let out_proj_w = t("out_proj.weight")?;
    let out_proj_b = t("out_proj.bias")?;
    let ssmax = if has_ssmax {
        Some(SsmaxW {
            base_fc1_w: t("ssmax_layer.base_mlp.0.weight")?,
            base_fc1_b: t("ssmax_layer.base_mlp.0.bias")?,
            base_fc2_w: t("ssmax_layer.base_mlp.2.weight")?,
            base_fc2_b: t("ssmax_layer.base_mlp.2.bias")?,
            query_fc1_w: t("ssmax_layer.query_mlp.0.weight")?,
            query_fc1_b: t("ssmax_layer.query_mlp.0.bias")?,
            query_fc2_w: t("ssmax_layer.query_mlp.2.weight")?,
            query_fc2_b: t("ssmax_layer.query_mlp.2.bias")?,
        })
    } else {
        None
    };
    Ok(AttnW { in_proj_w, in_proj_b, out_proj_w, out_proj_b, ssmax })
}

fn load_block(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    device: &Device,
    has_ssmax: bool,
) -> Result<BlockW> {
    let norm1_w = load_t(content, reader, &format!("{prefix}.norm1.weight"), device)?;
    let norm1_b = load_t(content, reader, &format!("{prefix}.norm1.bias"), device)?;
    let norm2_w = load_t(content, reader, &format!("{prefix}.norm2.weight"), device)?;
    let norm2_b = load_t(content, reader, &format!("{prefix}.norm2.bias"), device)?;
    let attn = load_attn(content, reader, &format!("{prefix}.attn"), device, has_ssmax)?;
    let fc1_w = load_t(content, reader, &format!("{prefix}.linear1.weight"), device)?;
    let fc1_b = load_t(content, reader, &format!("{prefix}.linear1.bias"), device)?;
    let fc2_w = load_t(content, reader, &format!("{prefix}.linear2.weight"), device)?;
    let fc2_b = load_t(content, reader, &format!("{prefix}.linear2.bias"), device)?;
    Ok(BlockW { norm1_w, norm1_b, norm2_w, norm2_b, attn, fc1_w, fc1_b, fc2_w, fc2_b })
}

impl TabIclModel {
    pub fn load(gguf_path: &Path, config: TabIclConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let col_in_linear_w = load_t(&content, &mut reader, "col_embedder.in_linear.weight", &device)?;
        let col_in_linear_b = load_t(&content, &mut reader, "col_embedder.in_linear.bias", &device)?;
        let col_y_w = load_t(&content, &mut reader, "col_embedder.y_encoder.weight", &device)?;
        let col_y_b = load_t(&content, &mut reader, "col_embedder.y_encoder.bias", &device)?;

        let mut col_blocks = Vec::with_capacity(config.col_num_blocks);
        for n in 0..config.col_num_blocks {
            let p = format!("col_embedder.tf_col.blocks.{n}");
            let ind_vectors = load_t(&content, &mut reader, &format!("{p}.ind_vectors"), &device)?;
            let attn1 = load_block(&content, &mut reader, &format!("{p}.multihead_attn1"), &device, true)?;
            let attn2 = load_block(&content, &mut reader, &format!("{p}.multihead_attn2"), &device, false)?;
            col_blocks.push(IsabW { ind_vectors, attn1, attn2 });
        }

        let cls_tokens = load_t(&content, &mut reader, "row_interactor.cls_tokens", &device)?;
        let row_out_ln_w = load_t(&content, &mut reader, "row_interactor.out_ln.weight", &device)?;
        let row_out_ln_b = load_t(&content, &mut reader, "row_interactor.out_ln.bias", &device)?;
        let rope_freqs_t = load_t(&content, &mut reader, "row_interactor.tf_row.rope.freqs", &device)?;
        let rope_freqs: Vec<f32> = rope_freqs_t.flatten_all()?.to_vec1()?;

        let mut row_blocks = Vec::with_capacity(config.row_num_blocks);
        for n in 0..config.row_num_blocks {
            let p = format!("row_interactor.tf_row.blocks.{n}");
            row_blocks.push(load_block(&content, &mut reader, &p, &device, false)?);
        }

        let icl_y_w = load_t(&content, &mut reader, "icl_predictor.y_encoder.weight", &device)?;
        let icl_y_b = load_t(&content, &mut reader, "icl_predictor.y_encoder.bias", &device)?;
        let icl_ln_w = load_t(&content, &mut reader, "icl_predictor.ln.weight", &device)?;
        let icl_ln_b = load_t(&content, &mut reader, "icl_predictor.ln.bias", &device)?;
        let decoder_fc1_w = load_t(&content, &mut reader, "icl_predictor.decoder.0.weight", &device)?;
        let decoder_fc1_b = load_t(&content, &mut reader, "icl_predictor.decoder.0.bias", &device)?;
        let decoder_fc2_w = load_t(&content, &mut reader, "icl_predictor.decoder.2.weight", &device)?;
        let decoder_fc2_b = load_t(&content, &mut reader, "icl_predictor.decoder.2.bias", &device)?;

        let mut icl_blocks = Vec::with_capacity(config.icl_num_blocks);
        for n in 0..config.icl_num_blocks {
            let p = format!("icl_predictor.tf_icl.blocks.{n}");
            icl_blocks.push(load_block(&content, &mut reader, &p, &device, true)?);
        }

        Ok(Self {
            device,
            config,
            col: ColEmbedderW {
                in_linear_w: col_in_linear_w,
                in_linear_b: col_in_linear_b,
                y_encoder_w: col_y_w,
                y_encoder_b: col_y_b,
                blocks: col_blocks,
            },
            row: RowInteractorW {
                cls_tokens,
                blocks: row_blocks,
                out_ln_w: row_out_ln_w,
                out_ln_b: row_out_ln_b,
                rope_freqs,
            },
            icl: IclPredictorW {
                y_encoder_w: icl_y_w,
                y_encoder_b: icl_y_b,
                blocks: icl_blocks,
                ln_w: icl_ln_w,
                ln_b: icl_ln_b,
                decoder_fc1_w,
                decoder_fc1_b,
                decoder_fc2_w,
                decoder_fc2_b,
            },
        })
    }

    // -----------------------------------------------------------------------
    // Prediction
    // -----------------------------------------------------------------------

    /// Zero-shot classification. `y_support` are class indices `0..n_classes` (must satisfy
    /// `n_classes <= 10` — the >10-class mixed-radix/hierarchical path is out of scope).
    /// Returns probabilities `[n_query][n_classes]`.
    pub fn predict_classification(
        &self,
        x_support: &[Vec<f32>],
        y_support: &[usize],
        x_query: &[Vec<f32>],
        n_classes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::ensure!(n_classes <= self.config.max_classes, "n_classes must be <= {}", self.config.max_classes);
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
        let x_t = Tensor::from_vec(flat, (h, t, group_size), &self.device)?;

        let y_onehot_col = self.onehot_linear(y_support, &self.col.y_encoder_w, &self.col.y_encoder_b)?;
        let col_out = self.col_embed_forward(&x_t, &y_onehot_col, train_size)?;
        let row_out = self.row_interact_forward(&col_out, train_size)?;
        let y_onehot_icl = self.onehot_linear(y_support, &self.icl.y_encoder_w, &self.icl.y_encoder_b)?;
        let logits = self.icl_forward(&row_out, &y_onehot_icl, train_size, n_classes)?;

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

    fn onehot_linear(&self, y: &[usize], w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let n = y.len();
        let num_classes = w.dim(1)?;
        let mut onehot = vec![0f32; n * num_classes];
        for (i, &c) in y.iter().enumerate() {
            onehot[i * num_classes + c] = 1.0;
        }
        let x = Tensor::from_vec(onehot, (n, num_classes), &self.device)?;
        zsfm_nn::linear_bias(&x, w, b).map_err(anyhow::Error::from)
    }

    /// `x_grouped`: `[H, T, group_size]`. `y_onehot`: `[train_size, embed_dim]` (already
    /// one-hot + linear-projected). Returns column embeddings `[H, T, embed_dim]`.
    fn col_embed_forward(&self, x_grouped: &Tensor, y_onehot: &Tensor, train_size: usize) -> Result<Tensor> {
        let src = zsfm_nn::linear_bias(x_grouped, &self.col.in_linear_w, &self.col.in_linear_b)?;
        let t = src.dim(1)?;
        let train_part = src.narrow(1, 0, train_size)?.broadcast_add(&y_onehot.unsqueeze(0)?)?;
        let mut src = if train_size < t {
            Tensor::cat(&[&train_part, &src.narrow(1, train_size, t - train_size)?], 1)?
        } else {
            train_part
        };

        let h = src.dim(0)?;
        for isab in &self.col.blocks {
            src = isab_forward(&src, isab, self.config.col_nhead, train_size, h)?;
        }
        Ok(src)
    }

    /// `col_embeddings`: `[H, T, embed_dim]`. Returns row representations `[T, icl_dim]`.
    fn row_interact_forward(&self, col_embeddings: &Tensor, _train_size: usize) -> Result<Tensor> {
        let h = col_embeddings.dim(0)?;
        let t = col_embeddings.dim(1)?;
        let dim = self.config.embed_dim;
        let n_cls = self.config.row_num_cls;

        let feat = col_embeddings.permute((1, 0, 2))?.contiguous()?; // (T, H, dim)
        let cls = self.row.cls_tokens.unsqueeze(0)?.expand((t, n_cls, dim))?.contiguous()?;
        let mut seq = Tensor::cat(&[&cls, &feat], 1)?; // (T, H+C, dim)

        let max_len = h + n_cls;
        let (cos, sin) = self.row_rope_table(max_len)?;

        let n_blocks = self.row.blocks.len();
        for (i, blk) in self.row.blocks.iter().enumerate() {
            if i + 1 == n_blocks {
                let cls_q = seq.narrow(1, 0, n_cls)?;
                seq = mha_block(&cls_q, &seq, &seq, blk, self.config.row_nhead, Some((&cos, &sin)), 0)?;
            } else {
                seq = mha_block(&seq, &seq, &seq, blk, self.config.row_nhead, Some((&cos, &sin)), 0)?;
            }
        }

        let out = zsfm_nn::layer_norm(&seq, &self.row.out_ln_w, &self.row.out_ln_b, LN_EPS)?; // (T, C, dim)
        out.reshape((t, n_cls * dim)).map_err(anyhow::Error::from)
    }

    fn row_rope_table(&self, max_len: usize) -> Result<(Tensor, Tensor)> {
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
        let cos_t = Tensor::from_vec(cos, (max_len, 2 * half), &self.device)?;
        let sin_t = Tensor::from_vec(sin, (max_len, 2 * half), &self.device)?;
        Ok((cos_t, sin_t))
    }

    /// `row_reprs`: `[T, icl_dim]`. `y_onehot`: `[train_size, icl_dim]`. Returns logits
    /// `[n_query, n_classes]` (already sliced to the query rows and the first `n_classes`
    /// output columns).
    fn icl_forward(&self, row_reprs: &Tensor, y_onehot: &Tensor, train_size: usize, n_classes: usize) -> Result<Tensor> {
        let t = row_reprs.dim(0)?;
        let train_part = row_reprs.narrow(0, 0, train_size)?.broadcast_add(y_onehot)?;
        let r = if train_size < t {
            Tensor::cat(&[&train_part, &row_reprs.narrow(0, train_size, t - train_size)?], 0)?
        } else {
            train_part
        };
        let mut r = r.unsqueeze(0)?; // (1, T, icl_dim) -- single table

        for blk in &self.icl.blocks {
            let kv = r.narrow(1, 0, train_size)?;
            r = mha_block(&r, &kv, &kv, blk, self.config.icl_nhead, None, train_size)?;
        }
        let r = r.squeeze(0)?; // (T, icl_dim)
        let r = zsfm_nn::layer_norm(&r, &self.icl.ln_w, &self.icl.ln_b, LN_EPS)?;
        let h = zsfm_nn::linear_bias(&r, &self.icl.decoder_fc1_w, &self.icl.decoder_fc1_b)?.gelu_erf()?;
        let logits = zsfm_nn::linear_bias(&h, &self.icl.decoder_fc2_w, &self.icl.decoder_fc2_b)?; // (T, max_classes)
        logits.narrow(0, train_size, t - train_size)?.narrow(1, 0, n_classes).map_err(anyhow::Error::from)
    }
}

/// One `InducedSelfAttentionBlock`: learned inducing points cross-attend to the *training*
/// rows (`attn1`, SSMax on this stage only), then the full column cross-attends back to the
/// refined inducing points (`attn2`). `src`: `[H, T, dim]`.
fn isab_forward(src: &Tensor, isab: &IsabW, n_heads: usize, train_size: usize, n_h: usize) -> Result<Tensor> {
    let num_inds = isab.ind_vectors.dim(0)?;
    let dim = isab.ind_vectors.dim(1)?;
    let ind = isab.ind_vectors.unsqueeze(0)?.expand((n_h, num_inds, dim))?.contiguous()?;

    let kv_train = src.narrow(1, 0, train_size)?;
    let hidden = mha_block(&ind, &kv_train, &kv_train, &isab.attn1, n_heads, None, train_size)?;
    mha_block(src, &hidden, &hidden, &isab.attn2, n_heads, None, num_inds)
}

/// One pre-norm `MultiheadAttentionBlock`: LN -> MHA (combined in-proj, optional RoPE, optional
/// SSMax) -> residual -> LN -> GELU-FFN -> residual. `q_in`/`k_in`/`v_in` are pre-selected by the
/// caller (always independently LayerNorm'd here — mathematically identical to the reference's
/// "reuse if same tensor" optimization, since LayerNorm is a pure deterministic function).
fn mha_block(
    q_in: &Tensor,
    k_in: &Tensor,
    v_in: &Tensor,
    blk: &BlockW,
    n_heads: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
    ssmax_n: usize,
) -> Result<Tensor> {
    let q_normed = zsfm_nn::layer_norm(q_in, &blk.norm1_w, &blk.norm1_b, LN_EPS)?;
    let k_normed = zsfm_nn::layer_norm(k_in, &blk.norm1_w, &blk.norm1_b, LN_EPS)?;
    let v_normed = zsfm_nn::layer_norm(v_in, &blk.norm1_w, &blk.norm1_b, LN_EPS)?;
    let attn_out = mha_combined(&q_normed, &k_normed, &v_normed, &blk.attn, n_heads, rope_cos_sin, ssmax_n)?;
    let x = (q_in + attn_out)?;
    let ff_in = zsfm_nn::layer_norm(&x, &blk.norm2_w, &blk.norm2_b, LN_EPS)?;
    let ff = zsfm_nn::linear_bias(&zsfm_nn::linear_bias(&ff_in, &blk.fc1_w, &blk.fc1_b)?.gelu_erf()?, &blk.fc2_w, &blk.fc2_b)?;
    Ok((x + ff)?)
}

/// Combined-in-projection multi-head attention (`nn.MultiheadAttention`-style: one packed
/// `in_proj_weight`/`bias` split into Q/K/V thirds). `q_in`/`k_in`/`v_in`: `[batch, seq, dim]`.
/// RoPE (applied to Q/K, row interactor only) and SSMax (applied to Q, column/ICL stages only)
/// never co-occur in this model.
fn mha_combined(
    q_in: &Tensor,
    k_in: &Tensor,
    v_in: &Tensor,
    attn: &AttnW,
    n_heads: usize,
    rope_cos_sin: Option<(&Tensor, &Tensor)>,
    ssmax_n: usize,
) -> Result<Tensor> {
    let dim = q_in.dim(2)?;
    let wq = attn.in_proj_w.narrow(0, 0, dim)?;
    let wk = attn.in_proj_w.narrow(0, dim, dim)?;
    let wv = attn.in_proj_w.narrow(0, 2 * dim, dim)?;
    let bq = attn.in_proj_b.narrow(0, 0, dim)?;
    let bk = attn.in_proj_b.narrow(0, dim, dim)?;
    let bv = attn.in_proj_b.narrow(0, 2 * dim, dim)?;

    let q = zsfm_nn::linear_bias(q_in, &wq, &bq)?;
    let k = zsfm_nn::linear_bias(k_in, &wk, &bk)?;
    let v = zsfm_nn::linear_bias(v_in, &wv, &bv)?;

    let batch = q.dim(0)?;
    let q_len = q.dim(1)?;
    let k_len = k.dim(1)?;
    let head_dim = dim / n_heads;

    let q = q.reshape((batch, q_len, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
    let k = k.reshape((batch, k_len, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
    let v = v.reshape((batch, k_len, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

    let (q, k) = if let Some((cos, sin)) = rope_cos_sin {
        (apply_rope(&q, cos, sin)?, apply_rope(&k, cos, sin)?)
    } else {
        (q, k)
    };

    let q = if let Some(ssmax) = &attn.ssmax {
        apply_ssmax(&q, ssmax, ssmax_n, n_heads, head_dim)?
    } else {
        q
    };

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v)?; // (batch, h, q_len, hd)
    let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((batch, q_len, dim))?;
    zsfm_nn::linear_bias(&out, &attn.out_proj_w, &attn.out_proj_b).map_err(anyhow::Error::from)
}

/// Non-interleaved RoPE (`rotate_half_contiguous`: split into first/second halves, negate and
/// swap). `x`: `[batch, heads, seq, head_dim]`. `cos`/`sin`: `[max_len, head_dim]`, narrowed
/// here to `x`'s own sequence length (so a Q slice starting at position 0 — e.g. the CLS-token
/// readout query — gets the matching leading rows of the table).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let seq = x.dim(2)?;
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;
    let cos = cos.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?; // (1,1,seq,hd)
    let sin = sin.narrow(0, 0, seq)?.unsqueeze(0)?.unsqueeze(0)?;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
}

/// Query-aware elementwise SSMax: `q * base_mlp(log n) * (1 + tanh(query_mlp(q)))`.
/// `q`: `[batch, n_heads, seq, head_dim]`.
fn apply_ssmax(q: &Tensor, ssmax: &SsmaxW, n: usize, n_heads: usize, head_dim: usize) -> Result<Tensor> {
    let device = q.device();
    let logn = (n.max(1) as f32).ln();
    let logn_t = Tensor::from_vec(vec![logn], (1, 1), device)?;
    let base = zsfm_nn::linear_bias(&logn_t, &ssmax.base_fc1_w, &ssmax.base_fc1_b)?.gelu_erf()?;
    let base = zsfm_nn::linear_bias(&base, &ssmax.base_fc2_w, &ssmax.base_fc2_b)?; // (1, n_heads*head_dim)
    let base = base.reshape((1, n_heads, 1, head_dim))?;

    let (batch, seq) = (q.dim(0)?, q.dim(2)?);
    let q_flat = q.reshape((batch * n_heads * seq, head_dim))?;
    let qm = zsfm_nn::linear_bias(&q_flat, &ssmax.query_fc1_w, &ssmax.query_fc1_b)?.gelu_erf()?;
    let qm = zsfm_nn::linear_bias(&qm, &ssmax.query_fc2_w, &ssmax.query_fc2_b)?;
    let modulation = (qm.tanh()? + 1.0)?.reshape((batch, n_heads, seq, head_dim))?;

    let scales = base.broadcast_mul(&modulation)?;
    Ok(q.broadcast_mul(&scales)?)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

// ---------------------------------------------------------------------------
// Preprocessing (mean-impute + standardize, fit on support/applied to both — same scope
// decision as Mitra/TabDPT; TabICL's own sklearn wrapper's fuller ensembling/normalizer-choice
// pipeline is not replicated)
// ---------------------------------------------------------------------------

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
        impute_mean[f] = if count > 0 { (sum / count as f64) as f32 } else { 0.0 };
    }
    let imputed: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| row.iter().enumerate().map(|(f, &v)| if v.is_nan() { impute_mean[f] } else { v }).collect())
        .collect();

    let mut mean = vec![0f32; n_feat];
    let mut std = vec![0f32; n_feat];
    for f in 0..n_feat {
        let m: f32 = imputed[..train_size].iter().map(|r| r[f]).sum::<f32>() / train_size as f32;
        let var: f32 = imputed[..train_size].iter().map(|r| (r[f] - m).powi(2)).sum::<f32>() / train_size as f32;
        mean[f] = m;
        std[f] = if var == 0.0 { 1.0 } else { var.sqrt() };
    }

    imputed.iter().map(|row| row.iter().enumerate().map(|(f, &v)| (v - mean[f]) / std[f]).collect()).collect()
}

/// Circular-permutation feature grouping (`col_feature_group="same"`): group `g`'s values are
/// the input features at offsets `2^0, 2^1, .., 2^(size-1)` past `g` (mod `H`) — note this does
/// *not* include feature `g` itself, only its power-of-two-shifted neighbors.
fn feature_group_same(rows: &[Vec<f32>], h: usize, size: usize) -> Vec<Vec<Vec<f32>>> {
    rows.iter()
        .map(|row| {
            (0..h)
                .map(|g| (0..size).map(|k| row[(g + (1usize << k)) % h]).collect())
                .collect()
        })
        .collect()
}
