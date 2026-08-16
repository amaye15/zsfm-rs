//! TabFM inference engine, with a real batch dimension `B` (`predict_batch`) — e.g. one batch
//! item per ensemble member, so `ensemble::orchestrate` can share the fixed cost of the 24-block
//! ICL stage across all members in one forward pass instead of paying it once per member. The
//! single-table `predict()` is a thin `B=1` wrapper around the same code path; none of the
//! attention/RMSNorm/RoPE math below changed to add batching — only the four "stage" functions
//! (`cell_embed`, `col_embedding_forward`, `row_interaction_forward`, `icl_forward`) gained a
//! leading `B` dimension, via reshapes around the *same* 3D attention calls they always made
//! (masks/weights are shared scalars across the batch — every member has the same row/feature
//! count, `train_size`, and `d`; only cell *values* and `cat_mask` vary per member).
//!
//! Architecture (from `tabfm/src/pytorch/model.py`, verified against the installed package):
//! CellEmbedder (per-cell grouped Fourier features + train-row y-embedding)
//!   -> ColEmbedding (SetTransformer / induced attention over rows, masked to train rows)
//!   -> prepend row_num_cls learned CLS tokens on the column axis
//!   -> RowInteraction (RoPE cross-column self-attention, masked to valid/unpadded columns)
//!   -> ColEmbedding (stage 2)
//!   -> RowInteraction (stage 2, output collapsed to the CLS-token slice -> icl_dim)
//!   -> ICLearning (24-block self-attention over rows, y re-injected at train rows, masked so
//!      only train rows are attendable keys) -> MLP decoder -> per-class logits or a scalar.
//!
//! Numeric details that matter for parity (see model-to-gguf skill's debugging ladder):
//! * RoPE frequencies are checkpoint-loaded buffers, never recomputed from a formula.
//! * `MultiheadAttention` pre-scales `q` by a learned per-dimension softplus'd scale, then calls
//!   attention with `scale=1.0` — do not apply an additional `1/sqrt(d)`.
//! * All masking is additive key-side masking (no causal masking anywhere).
//! * RoPE here is the *interleaved-pair* variant (`x[0::2]`, `x[1::2]`), NOT the Llama
//!   rotate-half variant used elsewhere in this repo (see `toto`'s `infer/rope.rs`) —
//!   deliberately not reused.

use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};
use zsfm_nn::{linear, load_weight};

use crate::config::TabFMConfig;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InferConfig {
    embed_dim: usize,
    max_classes: usize,
    col_num_blocks: usize,
    col_nhead: usize,
    /// Kept for config-shape parity with `config.json`; the SetTransformer's inducing-point
    /// count is implied by the loaded `ind_vectors` tensor's own shape, not read from here.
    #[allow(dead_code)]
    col_num_inds: usize,
    row_num_blocks: usize,
    row_nhead: usize,
    row_num_cls: usize,
    icl_num_blocks: usize,
    icl_nhead: usize,
    ff_factor: usize,
    feature_group_size: usize,
    num_freq: usize,
    norm_eps: f64,
    is_classifier: bool,
    /// `None` means "use the `TabFM.__init__` default of `icl_dim * 2`".
    decoder_hidden: Option<usize>,
}

/// Every field here is load-bearing architecture metadata (must match the checkpoint the GGUF
/// was converted from) — there is no sensible standalone `Default`. Build one from a parsed
/// `config.json` via `From<&TabFMConfig>` (or `TabFMModelBuilder::config_from`), then optionally
/// layer on the few genuinely optional overrides below.
impl From<&TabFMConfig> for InferConfig {
    fn from(tc: &TabFMConfig) -> Self {
        InferConfig {
            embed_dim: tc.embed_dim as usize,
            max_classes: tc.max_classes as usize,
            col_num_blocks: tc.col_num_blocks as usize,
            col_nhead: tc.col_nhead as usize,
            col_num_inds: tc.col_num_inds as usize,
            row_num_blocks: tc.row_num_blocks as usize,
            row_nhead: tc.row_nhead as usize,
            row_num_cls: tc.row_num_cls as usize,
            icl_num_blocks: tc.icl_num_blocks as usize,
            icl_nhead: tc.icl_nhead as usize,
            ff_factor: tc.ff_factor as usize,
            feature_group_size: tc.feature_group_size as usize,
            num_freq: tc.num_freq as usize,
            norm_eps: tc.norm_eps,
            is_classifier: tc.is_classifier,
            decoder_hidden: tc.decoder_hidden.map(|v| v as usize),
        }
    }
}

impl InferConfig {
    fn col_dim_ff(&self) -> usize { self.embed_dim * self.ff_factor }
    fn icl_dim(&self) -> usize { self.embed_dim * self.row_num_cls }
    fn icl_dim_ff(&self) -> usize { self.icl_dim() * self.ff_factor }
    fn decoder_hidden(&self) -> usize { self.decoder_hidden.unwrap_or(self.icl_dim() * 2) }
    fn out_dim(&self) -> usize { if self.is_classifier { self.max_classes } else { 1 } }

    // -- builder-style overrides of the few genuinely optional fields ---------

    pub fn with_decoder_hidden(mut self, v: Option<usize>) -> Self { self.decoder_hidden = v; self }
    pub fn with_norm_eps(mut self, v: f64) -> Self { self.norm_eps = v; self }
    pub fn with_num_freq(mut self, v: usize) -> Self { self.num_freq = v; self }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent constructor for [`TabFMModel`]: point it at a GGUF file and a config (from a parsed
/// `config.json` via [`config_from`](TabFMModelBuilder::config_from), or a hand-built
/// [`InferConfig`] via [`config`](TabFMModelBuilder::config)), then call
/// [`build`](TabFMModelBuilder::build).
///
/// ```no_run
/// use zsfm_tabfm::{TabFMConfig, TabFMModel};
///
/// # fn main() -> anyhow::Result<()> {
/// let tc = TabFMConfig::from_json(&std::fs::read_to_string("config.json")?)?;
/// let model = TabFMModel::builder("tabfm.gguf").config_from(&tc).build()?;
/// # Ok(()) }
/// ```
pub struct TabFMModelBuilder {
    gguf_path: PathBuf,
    config: Option<InferConfig>,
}

impl TabFMModelBuilder {
    fn new(gguf_path: impl Into<PathBuf>) -> Self {
        Self { gguf_path: gguf_path.into(), config: None }
    }

    /// Use an already-built [`InferConfig`] (e.g. `InferConfig::from(&tc).with_norm_eps(...)`).
    pub fn config(mut self, config: InferConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Map a parsed `config.json` (`TabFMConfig`) onto the model's `InferConfig`.
    pub fn config_from(mut self, tc: &TabFMConfig) -> Self {
        self.config = Some(InferConfig::from(tc));
        self
    }

    pub fn build(self) -> Result<TabFMModel> {
        let config = self
            .config
            .context("TabFMModelBuilder: no config set — call .config(...) or .config_from(...)")?;
        TabFMModel::load(&self.gguf_path, config)
    }
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Weights for one `MultiheadAttentionBlock` (attention sublayer + SwiGLU FFN sublayer, each
/// with its own pre/post RMSNorm).
struct MabWeights {
    q_w: Tensor, q_b: Tensor,
    k_w: Tensor, k_b: Tensor,
    v_w: Tensor, v_b: Tensor,
    o_w: Tensor, o_b: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    per_dim_scale: Tensor,
    pre_attn_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ff_norm: Tensor,
    post_ff_norm: Tensor,
    ffn_up_w: Tensor, ffn_up_b: Tensor,
    ffn_gate_w: Tensor, ffn_gate_b: Tensor,
    ffn_down_w: Tensor, ffn_down_b: Tensor,
}

/// One `InducedSelfAttentionBlock`: shared induced vectors + two `MultiheadAttentionBlock`s.
struct InducedBlockWeights {
    ind_vectors: Tensor, // [num_inds, E]
    mab1: MabWeights,
    mab2: MabWeights,
}

struct ColStackWeights {
    blocks: Vec<InducedBlockWeights>,
    out_w_w: Tensor, out_w_b: Tensor,
    out_norm_w: Tensor,
}

struct RowStackWeights {
    rope_freqs: Tensor, // [head_dim/2]
    blocks: Vec<MabWeights>,
    out_norm_w: Tensor,
}

/// A generic (weight, bias) linear-layer stack, GELU-tanh between layers (not after the last).
struct MlpWeights {
    layers: Vec<(Tensor, Tensor)>,
}

enum YEmbedWeights {
    /// classification: plain `nn.Embedding` lookup table `[max_classes, E]`.
    Embedding(Tensor),
    /// regression: 2-layer MLP `1 -> 6 -> E`.
    Mlp(MlpWeights),
}

enum YEncoderWeights {
    /// classification: `OneHotAndLinear` projection `[icl_dim, max_classes]` + bias.
    OneHot { proj_w: Tensor, proj_b: Tensor },
    /// regression: 2-layer MLP `1 -> decoder_hidden -> icl_dim`.
    Mlp(MlpWeights),
}

struct CellWeights {
    fourier_freq: Tensor,     // [feature_group_size, num_freq]
    fourier_freq_cat: Tensor, // [feature_group_size, num_freq]
    in_linear_w: Tensor, in_linear_b: Tensor,         // [E, 2*num_freq], [E]
    in_linear_cat_w: Tensor, in_linear_cat_b: Tensor, // [E, 2*num_freq], [E]
    y_embed: YEmbedWeights,
}

struct IclWeights {
    blocks: Vec<MabWeights>,
    out_norm_w: Tensor,
    y_encoder: YEncoderWeights,
    decoder: MlpWeights,
}

pub struct TabFMModel {
    device: Device,
    config: InferConfig,
    cell: CellWeights,
    colenc1: ColStackWeights,
    colenc2: ColStackWeights,
    rowenc1: RowStackWeights,
    rowenc2: RowStackWeights,
    cls_tokens: Tensor, // [row_num_cls, E]
    icl: IclWeights,
}

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

fn load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

fn load_mab(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    e: usize,
    ff: usize,
    device: &Device,
) -> Result<MabWeights> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(MabWeights {
        q_w: load_weight(content, reader, &p("attn_q.weight"), e, device)?,
        q_b: load_tensor(content, reader, &p("attn_q.bias"), device)?,
        k_w: load_weight(content, reader, &p("attn_k.weight"), e, device)?,
        k_b: load_tensor(content, reader, &p("attn_k.bias"), device)?,
        v_w: load_weight(content, reader, &p("attn_v.weight"), e, device)?,
        v_b: load_tensor(content, reader, &p("attn_v.bias"), device)?,
        o_w: load_weight(content, reader, &p("attn_o.weight"), e, device)?,
        o_b: load_tensor(content, reader, &p("attn_o.bias"), device)?,
        q_norm: load_tensor(content, reader, &p("q_norm.weight"), device)?,
        k_norm: load_tensor(content, reader, &p("k_norm.weight"), device)?,
        per_dim_scale: load_tensor(content, reader, &p("per_dim_scale"), device)?,
        pre_attn_norm: load_tensor(content, reader, &p("pre_attn_norm.weight"), device)?,
        post_attn_norm: load_tensor(content, reader, &p("post_attn_norm.weight"), device)?,
        pre_ff_norm: load_tensor(content, reader, &p("pre_ff_norm.weight"), device)?,
        post_ff_norm: load_tensor(content, reader, &p("post_ff_norm.weight"), device)?,
        ffn_up_w: load_weight(content, reader, &p("ffn_up.weight"), ff, device)?,
        ffn_up_b: load_tensor(content, reader, &p("ffn_up.bias"), device)?,
        ffn_gate_w: load_weight(content, reader, &p("ffn_gate.weight"), ff, device)?,
        ffn_gate_b: load_tensor(content, reader, &p("ffn_gate.bias"), device)?,
        ffn_down_w: load_weight(content, reader, &p("ffn_down.weight"), e, device)?,
        ffn_down_b: load_tensor(content, reader, &p("ffn_down.bias"), device)?,
    })
}

fn load_col_stack(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    stack: &str,
    num_blocks: usize,
    e: usize,
    ff: usize,
    device: &Device,
) -> Result<ColStackWeights> {
    let mut blocks = Vec::with_capacity(num_blocks);
    for n in 0..num_blocks {
        let ind_vectors = load_tensor(content, reader, &format!("{stack}.blk.{n}.ind_vectors"), device)?;
        let mab1 = load_mab(content, reader, &format!("{stack}.blk.{n}.mab1"), e, ff, device)?;
        let mab2 = load_mab(content, reader, &format!("{stack}.blk.{n}.mab2"), e, ff, device)?;
        blocks.push(InducedBlockWeights { ind_vectors, mab1, mab2 });
    }
    Ok(ColStackWeights {
        blocks,
        out_w_w: load_weight(content, reader, &format!("{stack}.out_w.weight"), e, device)?,
        out_w_b: load_tensor(content, reader, &format!("{stack}.out_w.bias"), device)?,
        out_norm_w: load_tensor(content, reader, &format!("{stack}.out_norm.weight"), device)?,
    })
}

fn load_row_stack(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    stack: &str,
    num_blocks: usize,
    e: usize,
    ff: usize,
    device: &Device,
) -> Result<RowStackWeights> {
    let rope_freqs = load_tensor(content, reader, &format!("{stack}.rope_freqs"), device)?;
    let mut blocks = Vec::with_capacity(num_blocks);
    for n in 0..num_blocks {
        blocks.push(load_mab(content, reader, &format!("{stack}.blk.{n}"), e, ff, device)?);
    }
    Ok(RowStackWeights {
        rope_freqs,
        blocks,
        out_norm_w: load_tensor(content, reader, &format!("{stack}.out_norm.weight"), device)?,
    })
}

fn load_mlp(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    prefix: &str,
    dims: &[usize], // e.g. [in, hidden, out] -> layers [(in,hidden), (hidden,out)]
    device: &Device,
) -> Result<MlpWeights> {
    let mut layers = Vec::with_capacity(dims.len() - 1);
    for i in 0..dims.len() - 1 {
        let out_dim = dims[i + 1];
        let w = load_weight(content, reader, &format!("{prefix}.mlp.{i}.weight"), out_dim, device)?;
        let b = load_tensor(content, reader, &format!("{prefix}.mlp.{i}.bias"), device)?;
        layers.push((w, b));
    }
    Ok(MlpWeights { layers })
}

impl TabFMModel {
    /// Start building a [`TabFMModel`] — see [`TabFMModelBuilder`].
    pub fn builder(gguf_path: impl Into<PathBuf>) -> TabFMModelBuilder {
        TabFMModelBuilder::new(gguf_path)
    }

    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        let device = Device::Cpu;
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let content = gguf_file::Content::read(&mut file).context("parse GGUF header")?;

        let e = config.embed_dim;
        let col_ff = config.col_dim_ff();
        let icl_dim = config.icl_dim();
        let icl_ff = config.icl_dim_ff();
        let decoder_hidden = config.decoder_hidden();
        let out_dim = config.out_dim();

        let cell = CellWeights {
            fourier_freq: load_tensor(&content, &mut file, "cell.fourier_freq", &device)?,
            fourier_freq_cat: load_tensor(&content, &mut file, "cell.fourier_freq_cat", &device)?,
            in_linear_w: load_weight(&content, &mut file, "cell.in_linear.weight", e, &device)?,
            in_linear_b: load_tensor(&content, &mut file, "cell.in_linear.bias", &device)?,
            in_linear_cat_w: load_weight(&content, &mut file, "cell.in_linear_cat.weight", e, &device)?,
            in_linear_cat_b: load_tensor(&content, &mut file, "cell.in_linear_cat.bias", &device)?,
            y_embed: if config.is_classifier {
                YEmbedWeights::Embedding(load_tensor(&content, &mut file, "cell.y_embed.weight", &device)?)
            } else {
                YEmbedWeights::Mlp(load_mlp(&content, &mut file, "cell.y_embed", &[1, 6, e], &device)?)
            },
        };

        let colenc1 = load_col_stack(&content, &mut file, "colenc1", config.col_num_blocks, e, col_ff, &device)?;
        let colenc2 = load_col_stack(&content, &mut file, "colenc2", config.col_num_blocks, e, col_ff, &device)?;
        let rowenc1 = load_row_stack(&content, &mut file, "rowenc1", config.row_num_blocks, e, col_ff, &device)?;
        let rowenc2 = load_row_stack(&content, &mut file, "rowenc2", config.row_num_blocks, e, col_ff, &device)?;

        let cls_tokens = load_tensor(&content, &mut file, "cls_tokens", &device)?;

        let mut icl_blocks = Vec::with_capacity(config.icl_num_blocks);
        for n in 0..config.icl_num_blocks {
            icl_blocks.push(load_mab(&content, &mut file, &format!("icl.blk.{n}"), icl_dim, icl_ff, &device)?);
        }
        let y_encoder = if config.is_classifier {
            YEncoderWeights::OneHot {
                proj_w: load_weight(&content, &mut file, "icl.y_encoder.projection.weight", icl_dim, &device)?,
                proj_b: load_tensor(&content, &mut file, "icl.y_encoder.projection.bias", &device)?,
            }
        } else {
            YEncoderWeights::Mlp(load_mlp(&content, &mut file, "icl.y_encoder", &[1, decoder_hidden, icl_dim], &device)?)
        };
        let decoder = load_mlp(&content, &mut file, "icl.decoder", &[icl_dim, decoder_hidden, out_dim], &device)?;
        let icl = IclWeights {
            blocks: icl_blocks,
            out_norm_w: load_tensor(&content, &mut file, "icl.out_norm.weight", &device)?,
            y_encoder,
            decoder,
        };

        Ok(Self { device, config, cell, colenc1, colenc2, rowenc1, rowenc2, cls_tokens, icl })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    pub fn is_classifier(&self) -> bool {
        self.config.is_classifier
    }

    /// Run one table (train rows followed by test rows) through TabFM. A thin `B=1` wrapper
    /// around `predict_batch` — see that method to run many tables (e.g. ensemble members) in
    /// one forward pass.
    ///
    /// `x`: `[T][H]` padded feature matrix (numeric; categorical columns are pre-encoded to
    /// floats by the caller). `y`: `[T]` labels (any finite placeholder at test-row positions is
    /// fine — it's masked internally and never influences the output). `train_size`: number of
    /// leading rows that are training rows. `cat_mask`: `[H]`, which columns are categorical
    /// (defaults to all-false). `d`: actual (unpadded) feature count (defaults to `H`).
    ///
    /// Returns `[T][out_dim]` raw logits (classification, `out_dim = max_classes`) or a
    /// `[T][1]` scalar (regression) — only rows `>= train_size` are meaningful predictions.
    pub fn predict(
        &self,
        x: &[Vec<f32>],
        y: &[f32],
        train_size: usize,
        cat_mask: Option<&[bool]>,
        d: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let h_len = x.first().map(|r| r.len()).unwrap_or(0);
        let default_mask = vec![false; h_len];
        let cat_mask = cat_mask.unwrap_or(&default_mask).to_vec();
        let out = self.predict_batch(&[x.to_vec()], &[y.to_vec()], train_size, &[cat_mask], d)?;
        out.into_iter().next().context("predict_batch returned no batch items")
    }

    /// Runs `B` independent tables through **one** forward pass, sharing the fixed cost of the
    /// 24-block ICL stage (and every other stage) across all of them instead of paying it once
    /// per table. Every table must share the same row count `T` and feature count `H`, and
    /// `train_size`/`d` are shared scalars across the whole batch — this holds for TabFM's
    /// ensemble members, which only differ in cell *values* (per-member feature
    /// permutation/scaling) and `cat_mask` (which position is categorical shifts with the
    /// permutation), never in table shape. Returns `[B][T][out_dim]`.
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
        anyhow::ensure!(t_len > 0, "x must have at least one row");
        let h_len = x_batch[0][0].len();
        for xb in x_batch {
            anyhow::ensure!(xb.len() == t_len, "all batch items must have the same row count");
            anyhow::ensure!(xb.iter().all(|r| r.len() == h_len), "all rows of x must have the same length");
        }
        anyhow::ensure!(y_batch.len() == b_len, "y_batch must have length B");
        anyhow::ensure!(y_batch.iter().all(|yb| yb.len() == t_len), "each y must have the same length as x's rows");
        anyhow::ensure!(train_size <= t_len, "train_size must be <= number of rows");
        anyhow::ensure!(cat_mask_batch.len() == b_len, "cat_mask_batch must have length B");
        anyhow::ensure!(cat_mask_batch.iter().all(|m| m.len() == h_len), "cat_mask must have length H");
        let d_val = d.unwrap_or(h_len).min(h_len);

        // 1. Cell embedding: [B, T, H, E]
        let emb0 = self.cell_embed(x_batch, y_batch, train_size, cat_mask_batch, d_val)?;

        // 2. Column embedding stage 1: [B, T, H, E]
        let emb1 = self.col_embedding_forward(&emb0, train_size, &self.colenc1)?;

        // 3. Prepend CLS tokens on the column axis: [B, T, row_num_cls + H, E]
        let num_cls = cfg.row_num_cls;
        let cls = self
            .cls_tokens
            .reshape((1, 1, num_cls, cfg.embed_dim))?
            .broadcast_as((b_len, t_len, num_cls, cfg.embed_dim))?
            .contiguous()?;
        let emb2 = Tensor::cat(&[&cls, &emb1], 2)?;

        // 4. Row interaction stage 1 (full output): [B, T, num_cls+H, E]
        let d_plus_cls = d_val + num_cls;
        let emb3 = self.row_interaction_forward(&emb2, d_plus_cls, &self.rowenc1, true)?;

        // 5. Column embedding stage 2: [B, T, num_cls+H, E]
        let emb4 = self.col_embedding_forward(&emb3, train_size, &self.colenc2)?;

        // 6. Row interaction stage 2 (CLS-only output): [B, T, icl_dim]
        let reps = self.row_interaction_forward(&emb4, d_plus_cls, &self.rowenc2, false)?;

        // 7. In-context learning: [B, T, out_dim]
        let logits = self.icl_forward(&reps, y_batch, train_size)?;

        let out_dim = cfg.out_dim();
        let flat: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let mut result = vec![vec![vec![0f32; out_dim]; t_len]; b_len];
        for (bb, batch_item) in result.iter_mut().enumerate() {
            for (t, row) in batch_item.iter_mut().enumerate() {
                let base = ((bb * t_len) + t) * out_dim;
                row.copy_from_slice(&flat[base..base + out_dim]);
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Stage 1: cell embedding (plain Rust — grouped Fourier features per cell)
    // -----------------------------------------------------------------------

    fn cell_embed(
        &self,
        x_batch: &[Vec<Vec<f32>>],
        y_batch: &[Vec<f32>],
        train_size: usize,
        cat_mask_batch: &[Vec<bool>],
        d: usize,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let b_len = x_batch.len();
        let t_len = x_batch[0].len();
        let h_len = x_batch[0][0].len();
        let fgs = cfg.feature_group_size;
        let num_freq = cfg.num_freq;
        let e = cfg.embed_dim;
        let d_safe = d.max(1);

        // group index table: idx[g][h] = (h + 2^g - 1) % d_safe (shared: d is a batch-wide scalar)
        let mut idx = vec![vec![0usize; h_len]; fgs];
        for (g, row) in idx.iter_mut().enumerate() {
            let offset = (1usize << g) - 1;
            for (h, slot) in row.iter_mut().enumerate() {
                *slot = (h + offset) % d_safe;
            }
        }

        let ff: Vec<f32> = self.cell.fourier_freq.flatten_all()?.to_vec1()?;
        let ffc: Vec<f32> = self.cell.fourier_freq_cat.flatten_all()?.to_vec1()?;
        let in_w: Vec<f32> = self.cell.in_linear_w.flatten_all()?.to_vec1()?;
        let in_b: Vec<f32> = self.cell.in_linear_b.to_vec1()?;
        let in_w_cat: Vec<f32> = self.cell.in_linear_cat_w.flatten_all()?.to_vec1()?;
        let in_b_cat: Vec<f32> = self.cell.in_linear_cat_b.to_vec1()?;

        let y_emb_t = compute_y_embed(y_batch, &self.cell.y_embed, cfg.max_classes, &self.device)?; // [B,T,E]
        let y_emb: Vec<f32> = y_emb_t.flatten_all()?.to_vec1()?; // [B*T*E]

        let mut out = vec![0f32; b_len * t_len * h_len * e];
        for bb in 0..b_len {
            let x = &x_batch[bb];
            let cat_mask = &cat_mask_batch[bb];
            for tt in 0..t_len {
                let add_y = tt < train_size;
                for hh in 0..h_len {
                    if hh >= d {
                        continue; // padded column: leave zeroed
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

        Ok(Tensor::from_vec(out, (b_len, t_len, h_len, e), &self.device)?)
    }

    // -----------------------------------------------------------------------
    // Stage 2/5: column embedding (SetTransformer, sequence axis = rows)
    // -----------------------------------------------------------------------

    fn col_embedding_forward(&self, x: &Tensor, train_size: usize, w: &ColStackWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let (b_len, t_len, hc, e) = x.dims4()?;
        // [B, T, HC, E] -> [B, HC, T, E] -> [(B*HC), T, E] (columns become the extended-batch
        // axis; the same reshape-around-unchanged-attention-code trick as before, now with a
        // real B folded into that batch axis alongside HC).
        let src = x.permute((0, 2, 1, 3))?.contiguous()?.reshape((b_len * hc, t_len, e))?;

        let mask = additive_key_mask(t_len, train_size, &self.device)?;

        let mut cur = src;
        for blk in &w.blocks {
            let n = cur.dim(0)?;
            let ind = blk
                .ind_vectors
                .unsqueeze(0)?
                .broadcast_as((n, blk.ind_vectors.dim(0)?, blk.ind_vectors.dim(1)?))?
                .contiguous()?;
            let hidden = mab_forward(&ind, &cur, &cur, &blk.mab1, cfg.col_nhead, cfg.norm_eps, Some(&mask), None)?;
            cur = mab_forward(&cur, &hidden, &hidden, &blk.mab2, cfg.col_nhead, cfg.norm_eps, None, None)?;
        }

        let projected = linear(&cur, &w.out_w_w, Some(&w.out_w_b))?;
        let normed = rms_norm(&projected, &w.out_norm_w, cfg.norm_eps)?;

        // [(B*HC), T, E] -> [B, HC, T, E] -> [B, T, HC, E]
        let out = normed.reshape((b_len, hc, t_len, e))?.permute((0, 2, 1, 3))?.contiguous()?;
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Stage 4/6: row interaction (RoPE self-attention, sequence axis = columns)
    // -----------------------------------------------------------------------

    fn row_interaction_forward(
        &self,
        x: &Tensor, // [B, T, HC, E]
        d_plus_cls: usize,
        w: &RowStackWeights,
        output_full: bool,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let (b_len, t_len, hc, e) = x.dims4()?;
        let mask = additive_key_mask(hc, d_plus_cls, &self.device)?;
        let rope = TabfmRope { freqs: w.rope_freqs.clone() };

        // [B, T, HC, E] -> [(B*T), HC, E]: B and T are both "extended batch" here (attention runs
        // over the HC/column axis), and are already contiguous/adjacent leading dims.
        let mut cur = x.reshape((b_len * t_len, hc, e))?;
        for blk in &w.blocks {
            cur = mab_forward(&cur, &cur, &cur, blk, cfg.row_nhead, cfg.norm_eps, Some(&mask), Some(&rope))?;
        }

        if output_full {
            let normed = rms_norm(&cur, &w.out_norm_w, cfg.norm_eps)?;
            Ok(normed.reshape((b_len, t_len, hc, e))?)
        } else {
            let num_cls = cfg.row_num_cls;
            let sliced = cur.narrow(1, 0, num_cls)?; // [(B*T), num_cls, E]
            let normed = rms_norm(&sliced, &w.out_norm_w, cfg.norm_eps)?;
            let icl_dim = num_cls * e;
            Ok(normed.reshape((b_len, t_len, icl_dim))?)
        }
    }

    // -----------------------------------------------------------------------
    // Stage 7: in-context learning
    // -----------------------------------------------------------------------

    fn icl_forward(&self, reps: &Tensor, y_batch: &[Vec<f32>], train_size: usize) -> Result<Tensor> {
        let cfg = &self.config;
        let (_b_len, t_len, _icl_dim) = reps.dims3()?; // reps: [B, T, icl_dim] — a real batch now,
        // unlike before where a synthetic B=1 was faked via unsqueeze/squeeze around this stage.

        let y_enc = compute_y_encoder(y_batch, &self.icl.y_encoder, cfg.max_classes, &self.device)?; // [B, T, icl_dim]
        let tm: Vec<f32> = (0..t_len).map(|t| if t < train_size { 1.0 } else { 0.0 }).collect();
        let tm_t = Tensor::from_vec(tm, (t_len, 1), &self.device)?;
        let r = (reps + y_enc.broadcast_mul(&tm_t)?)?; // [B, T, icl_dim]

        let mask = additive_key_mask(t_len, train_size, &self.device)?;

        let mut cur = r;
        for blk in &self.icl.blocks {
            cur = mab_forward(&cur, &cur, &cur, blk, cfg.icl_nhead, cfg.norm_eps, Some(&mask), None)?;
        }
        let normed = rms_norm(&cur, &self.icl.out_norm_w, cfg.norm_eps)?;
        mlp_forward(&normed, &self.icl.decoder)
    }
}

// ---------------------------------------------------------------------------
// y-embedding / y-encoder helpers
// ---------------------------------------------------------------------------

fn compute_y_embed(y_batch: &[Vec<f32>], w: &YEmbedWeights, max_classes: usize, device: &Device) -> Result<Tensor> {
    let b_len = y_batch.len();
    let t = y_batch[0].len();
    match w {
        YEmbedWeights::Embedding(emb_w) => {
            let e = emb_w.dim(1)?;
            let emb: Vec<f32> = emb_w.flatten_all()?.to_vec1()?;
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
            Ok(Tensor::from_vec(out, (b_len, t, e), device)?)
        }
        YEmbedWeights::Mlp(mlp) => {
            let flat: Vec<f32> = y_batch.iter().flatten().copied().collect();
            let y_col = Tensor::from_vec(flat, (b_len * t, 1), device)?;
            let out = mlp_forward(&y_col, mlp)?;
            let e = out.dim(1)?;
            Ok(out.reshape((b_len, t, e))?)
        }
    }
}

fn compute_y_encoder(y_batch: &[Vec<f32>], w: &YEncoderWeights, max_classes: usize, device: &Device) -> Result<Tensor> {
    let b_len = y_batch.len();
    let t = y_batch[0].len();
    match w {
        YEncoderWeights::OneHot { proj_w, proj_b } => {
            let icl_dim = proj_w.dim(0)?;
            let w_flat: Vec<f32> = proj_w.flatten_all()?.to_vec1()?; // [icl_dim, max_classes] row-major
            let bias: Vec<f32> = proj_b.to_vec1()?;
            let mut out = vec![0f32; b_len * t * icl_dim];
            for (bb, y) in y_batch.iter().enumerate() {
                for i in 0..t {
                    let yi = y[i] as i64;
                    let valid = yi >= 0 && (yi as usize) < max_classes;
                    let dst = (bb * t + i) * icl_dim;
                    for e in 0..icl_dim {
                        let w_contrib = if valid { w_flat[e * max_classes + yi as usize] } else { 0.0 };
                        out[dst + e] = bias[e] + w_contrib;
                    }
                }
            }
            Ok(Tensor::from_vec(out, (b_len, t, icl_dim), device)?)
        }
        YEncoderWeights::Mlp(mlp) => {
            let flat: Vec<f32> = y_batch.iter().flatten().copied().collect();
            let y_col = Tensor::from_vec(flat, (b_len * t, 1), device)?;
            let out = mlp_forward(&y_col, mlp)?;
            let icl_dim = out.dim(1)?;
            Ok(out.reshape((b_len, t, icl_dim))?)
        }
    }
}

// ---------------------------------------------------------------------------
// Generic tensor ops
// ---------------------------------------------------------------------------

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    zsfm_nn::rms_norm(&x.to_dtype(DType::F32)?, Some(w), eps)
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    Ok(((x.neg()?.exp()? + 1.0)?).recip()?)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    Ok((x * sigmoid(x)?)?)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok((x.exp()? + 1.0)?.log()?)
}

fn mlp_forward(x: &Tensor, w: &MlpWeights) -> Result<Tensor> {
    let n = w.layers.len();
    let mut h = x.clone();
    for (i, (lw, lb)) in w.layers.iter().enumerate() {
        h = linear(&h, lw, Some(lb))?;
        if i < n - 1 {
            h = h.gelu()?;
        }
    }
    Ok(h)
}

/// Additive attention mask over `[0, seq_len)` keys, valid where `key_idx < valid_len`.
/// Shape `[1, 1, 1, seq_len]`, broadcastable against `[N, nhead, Sq, seq_len]` attention scores.
fn additive_key_mask(seq_len: usize, valid_len: usize, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..seq_len)
        .map(|i| if i < valid_len { 0.0 } else { -1e9 })
        .collect();
    Ok(Tensor::from_vec(data, (1, 1, 1, seq_len), device)?)
}

// ---------------------------------------------------------------------------
// Interleaved-pair RoPE (checkpoint-loaded frequencies — never recomputed)
// ---------------------------------------------------------------------------

struct TabfmRope {
    freqs: Tensor, // [head_dim/2]
}

impl TabfmRope {
    /// Rotate `x: [N, T, nhead, head_dim]` over its `T` (dim 1) axis.
    fn rotate(&self, x: &Tensor) -> Result<Tensor> {
        let (_n, t, _nh, hd) = x.dims4()?;
        let half = hd / 2;
        let device = x.device();

        let positions: Vec<f32> = (0..t).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(positions, (t,), device)?;
        let freqs = self.freqs.to_dtype(DType::F32)?; // [half]
        let f = pos.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?; // [t, half]
        let cos = f.cos()?;
        let sin = f.sin()?;
        // repeat_interleave(2, -1): stack + reshape duplicates each element into adjacent pairs.
        let cos_i = Tensor::stack(&[&cos, &cos], 2)?.reshape((t, hd))?.reshape((1, t, 1, hd))?;
        let sin_i = Tensor::stack(&[&sin, &sin], 2)?.reshape((t, hd))?.reshape((1, t, 1, hd))?;

        let x_pairs = x.reshape((x.dim(0)?, t, x.dim(2)?, half, 2))?;
        let x1 = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // even indices
        let x2 = x_pairs.narrow(4, 1, 1)?.squeeze(4)?; // odd indices
        let rot = Tensor::stack(&[&x2.neg()?, &x1], 4)?.reshape(x.shape())?;

        Ok((x.broadcast_mul(&cos_i)? + rot.broadcast_mul(&sin_i)?)?)
    }
}

// ---------------------------------------------------------------------------
// MultiheadAttentionBlock forward (attention sublayer + SwiGLU FFN sublayer)
// ---------------------------------------------------------------------------

fn mab_forward(
    q_raw: &Tensor,
    k_raw: &Tensor,
    v_raw: &Tensor,
    w: &MabWeights,
    nhead: usize,
    eps: f64,
    mask: Option<&Tensor>,
    rope: Option<&TabfmRope>,
) -> Result<Tensor> {
    let qn = rms_norm(q_raw, &w.pre_attn_norm, eps)?;
    let kn = rms_norm(k_raw, &w.pre_attn_norm, eps)?;
    let vn = rms_norm(v_raw, &w.pre_attn_norm, eps)?;

    let attn_out = mha_core(&qn, &kn, &vn, w, nhead, eps, mask, rope)?;
    let a = rms_norm(&attn_out, &w.post_attn_norm, eps)?;
    let x = (q_raw + a)?;

    let xn = rms_norm(&x, &w.pre_ff_norm, eps)?;
    let gate = silu(&linear(&xn, &w.ffn_gate_w, Some(&w.ffn_gate_b))?)?;
    let up = linear(&xn, &w.ffn_up_w, Some(&w.ffn_up_b))?;
    let ff = linear(&(gate * up)?, &w.ffn_down_w, Some(&w.ffn_down_b))?;
    let ff = rms_norm(&ff, &w.post_ff_norm, eps)?;

    Ok((x + ff)?)
}

fn mha_core(
    qn: &Tensor,
    kn: &Tensor,
    vn: &Tensor,
    w: &MabWeights,
    nhead: usize,
    eps: f64,
    mask: Option<&Tensor>,
    rope: Option<&TabfmRope>,
) -> Result<Tensor> {
    let (n, sq, e) = qn.dims3()?;
    let sk = kn.dim(1)?;
    let hd = e / nhead;

    let q = linear(qn, &w.q_w, Some(&w.q_b))?.reshape((n, sq, nhead, hd))?;
    let k = linear(kn, &w.k_w, Some(&w.k_b))?.reshape((n, sk, nhead, hd))?;
    let v = linear(vn, &w.v_w, Some(&w.v_b))?.reshape((n, sk, nhead, hd))?;

    let (q, k) = match rope {
        Some(r) => (r.rotate(&q)?, r.rotate(&k)?),
        None => (q, k),
    };

    let q = rms_norm(&q, &w.q_norm, eps)?;
    let k = rms_norm(&k, &w.k_norm, eps)?;

    // scale = log2(e) / sqrt(hd) * softplus(per_dim_scale); attention itself uses scale=1.0.
    let scale = (softplus(&w.per_dim_scale)? * (1.442695041_f64 / (hd as f64).sqrt()))?;
    let q = q.broadcast_mul(&scale)?;

    let q = q.permute((0, 2, 1, 3))?.contiguous()?; // [N, nhead, Sq, hd]
    let k = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v = v.permute((0, 2, 1, 3))?.contiguous()?;

    let mut scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?; // [N, nhead, Sq, Sk]
    if let Some(m) = mask {
        scores = scores.broadcast_add(m)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v)?; // [N, nhead, Sq, hd]
    let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((n, sq, e))?;

    linear(&out, &w.o_w, Some(&w.o_b))
}

#[cfg(test)]
mod tests {
    use super::TabFMModel;

    /// Compile-time check that `TabFMModel` (and every `Tensor` it holds, CPU backend) is safe
    /// to share as `&TabFMModel` across threads — required for parallelizing the ensemble-member
    /// loop in `ensemble::orchestrate` over a shared, read-only model reference.
    #[test]
    fn test_model_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TabFMModel>();
    }
}
