//! Lag-Llama inference engine with KV caching.
//!
//! Two-phase inference:
//! 1. Prefill: full forward pass over context tokens, collects K/V cache per layer.
//! 2. Decode:  per-step single-token forward pass in pure Rust (zero Candle overhead).

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};
use simdeez::prelude::*;

use crate::config::LagLlamaConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct TransformerBlock {
    rms1_w: Tensor,
    rms2_w: Tensor,
    qkv_w:  Tensor,  // fused [3*n_embd, n_embd]
    c_w:    Tensor,
    fc1_w:  Tensor,
    fc2_w:  Tensor,
    proj_w: Tensor,
}

struct RawBlockWeights {
    rms1: Vec<f32>,  // [n_embd]
    rms2: Vec<f32>,  // [n_embd]
    qkv:  Vec<f32>,  // [3*n_embd, n_embd] row-major
    c:    Vec<f32>,  // [n_embd, n_embd]
    fc1:  Vec<f32>,  // [mlp_hidden, n_embd]
    fc2:  Vec<f32>,  // [mlp_hidden, n_embd]
    proj: Vec<f32>,  // [n_embd, mlp_hidden]
}

pub struct LagLlamaModel {
    device: Device,
    config: LagLlamaConfig,
    rope_cos: Tensor,
    rope_sin: Tensor,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
    wte_w: Tensor,
    wte_b: Tensor,
    blocks: Vec<TransformerBlock>,
    // Raw arrays for zero-overhead decode loop
    raw_blocks:   Vec<RawBlockWeights>,
    rope_cos_raw: Vec<f32>,  // [max_pos * half_head_dim]
    rope_sin_raw: Vec<f32>,
    norm_f_raw:   Vec<f32>,  // [n_embd]
    wte_w_raw:    Vec<f32>,  // [n_embd, feature_size]
    wte_b_raw:    Vec<f32>,  // [n_embd]
    mu_w_raw:     Vec<f32>,  // [n_embd] (flattened from head weight)
    mu_b_raw:     Vec<f32>,  // [1]
}

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

fn load_t(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    zsfm_nn::load_tensor(content, reader, name, device, DType::F32)
}

impl LagLlamaModel {
    pub fn load(gguf_path: &Path, config: LagLlamaConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let wte_w = load_t(&content, &mut reader, "enc.wte.weight", &device)?;
        let wte_b = load_t(&content, &mut reader, "enc.wte.bias", &device)?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for n in 0..config.n_layer {
            let p = |s: &str| format!("blk.{n}.{s}");
            let q_w  = load_t(&content, &mut reader, &p("attn_q.weight"),  &device)?;
            let kv_w = load_t(&content, &mut reader, &p("attn_kv.weight"), &device)?;
            blocks.push(TransformerBlock {
                rms1_w: load_t(&content, &mut reader, &p("rms1.weight"),    &device)?,
                rms2_w: load_t(&content, &mut reader, &p("rms2.weight"),    &device)?,
                qkv_w:  Tensor::cat(&[&q_w, &kv_w], 0)?,
                c_w:    load_t(&content, &mut reader, &p("attn_c.weight"),  &device)?,
                fc1_w:  load_t(&content, &mut reader, &p("mlp_fc1.weight"), &device)?,
                fc2_w:  load_t(&content, &mut reader, &p("mlp_fc2.weight"), &device)?,
                proj_w: load_t(&content, &mut reader, &p("mlp_proj.weight"), &device)?,
            });
        }

        let norm_f_w = load_t(&content, &mut reader, "norm_f.weight",   &device)?;
        let mu_w_t   = load_t(&content, &mut reader, "head.mu.weight",   &device)?;
        let mu_b_t   = load_t(&content, &mut reader, "head.mu.bias",     &device)?;

        let head_dim = config.n_embd_per_head;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let max_pos = config.max_context_length + 4096;
        let mut cos_vals = vec![0.0f32; max_pos * half];
        let mut sin_vals = vec![0.0f32; max_pos * half];
        for p in 0..max_pos {
            let pos = p as f32;
            for i in 0..half {
                let theta = pos * inv_freq[i];
                cos_vals[p * half + i] = theta.cos();
                sin_vals[p * half + i] = theta.sin();
            }
        }

        // Keep raw copies before Tensor::from_vec consumes the Vecs
        let rope_cos_raw = cos_vals.clone();
        let rope_sin_raw = sin_vals.clone();
        let rope_cos = Tensor::from_vec(cos_vals, (max_pos, half), &device)?;
        let rope_sin = Tensor::from_vec(sin_vals, (max_pos, half), &device)?;

        // Extract global raw weights
        let norm_f_raw = norm_f_w.flatten_all()?.to_vec1::<f32>()?;
        let wte_w_raw  = wte_w.flatten_all()?.to_vec1::<f32>()?;
        let wte_b_raw  = wte_b.flatten_all()?.to_vec1::<f32>()?;
        let mu_w_raw   = mu_w_t.flatten_all()?.to_vec1::<f32>()?;
        let mu_b_raw   = mu_b_t.flatten_all()?.to_vec1::<f32>()?;

        // Extract per-block raw weights
        let mut raw_blocks = Vec::with_capacity(config.n_layer);
        for blk in &blocks {
            raw_blocks.push(RawBlockWeights {
                rms1: blk.rms1_w.flatten_all()?.to_vec1::<f32>()?,
                rms2: blk.rms2_w.flatten_all()?.to_vec1::<f32>()?,
                qkv:  blk.qkv_w.flatten_all()?.to_vec1::<f32>()?,
                c:    blk.c_w.flatten_all()?.to_vec1::<f32>()?,
                fc1:  blk.fc1_w.flatten_all()?.to_vec1::<f32>()?,
                fc2:  blk.fc2_w.flatten_all()?.to_vec1::<f32>()?,
                proj: blk.proj_w.flatten_all()?.to_vec1::<f32>()?,
            });
        }

        Ok(Self {
            device,
            config,
            rope_cos,
            rope_sin,
            causal_mask_cache: Mutex::new(HashMap::new()),
            wte_w,
            wte_b,
            blocks,
            raw_blocks,
            rope_cos_raw,
            rope_sin_raw,
            norm_f_raw,
            wte_w_raw,
            wte_b_raw,
            mu_w_raw,
            mu_b_raw,
        })
    }

    // -----------------------------------------------------------------------
    // Forecasting: Candle prefill + raw-array decode loop
    // -----------------------------------------------------------------------

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let max_lag = *cfg.lags_seq.iter().max().unwrap_or(&0);

        let (loc, scale) = robust_stats(context);
        let scale = scale.max(1e-8);

        let mut hist: Vec<f32> = vec![0.0; max_lag + 1];
        for &v in context {
            hist.push((v - loc) / scale);
        }

        let ctx_buf_end   = hist.len();
        let ctx_buf_start = ctx_buf_end.saturating_sub(cfg.max_context_length);
        let seq_len       = ctx_buf_end - ctx_buf_start;

        // --- Phase 1: Candle prefill (full sequence, once) ---
        let feat_ctx = build_feature_matrix(&hist, ctx_buf_start, seq_len, cfg);
        let x = Tensor::from_vec(feat_ctx, (seq_len, cfg.feature_size), &self.device)?;
        let mut h = zsfm_nn::linear_bias(&x, &self.wte_w, &self.wte_b)?;

        let mut kv_caches: Vec<(Tensor, Tensor)> = Vec::with_capacity(cfg.n_layer);
        for blk in &self.blocks {
            let (h_out, k, v) = self.prefill_block(&h, blk, seq_len, ctx_buf_start)?;
            h = h_out;
            kv_caches.push((k, v));
        }

        // Extract last hidden state + KV caches into raw arrays (one-time cost)
        let mut h_raw: Vec<f32> = h.get(seq_len - 1)?.to_vec1()?;

        let n_head     = cfg.n_head;
        let head_dim   = cfg.n_embd_per_head;
        let n_embd     = cfg.n_embd;
        let mlp_hidden = cfg.mlp_hidden;
        let feat_size  = cfg.feature_size;

        // Per-layer, per-head KV buffers; pre-reserve full decode capacity
        let mut kv_raw = extract_kv_caches_raw(&kv_caches, n_head, head_dim, horizon)?;
        let mut kv_len = seq_len;

        // --- Phase 2: pure-Rust decode loop (zero Candle ops per step) ---
        let max_kv_len = seq_len + horizon;
        let mut h_tmp          = vec![0.0f32; n_embd];
        let mut qkv_buf        = vec![0.0f32; 3 * n_embd];
        let mut proj_buf       = vec![0.0f32; n_embd];
        let mut mlp_gate       = vec![0.0f32; mlp_hidden];
        let mut mlp_up         = vec![0.0f32; mlp_hidden];
        let mut mlp_out        = vec![0.0f32; n_embd];
        let mut attn_out       = vec![0.0f32; n_embd];
        let mut scores_scratch = vec![0.0f32; n_head * max_kv_len];

        let mut preds = Vec::with_capacity(horizon);
        let mut rope_offset = ctx_buf_start + seq_len;

        for step in 0..horizon {
            // Predict from current hidden state
            h_tmp.copy_from_slice(&h_raw);
            rms_norm_raw(&mut h_tmp, &self.norm_f_raw, 1e-5);
            let pred_scaled = raw_dot(&h_tmp, &self.mu_w_raw) + self.mu_b_raw[0];
            preds.push(pred_scaled * scale + loc);

            if step == horizon - 1 { break; }

            // Append normalized prediction to history
            hist.push(pred_scaled);

            // Build feature for next token and embed it
            let abs_t = hist.len() - 1;
            let feat_one = build_one_feature(&hist, abs_t, cfg);
            raw_gemv_bias(&feat_one, &self.wte_w_raw, &self.wte_b_raw,
                          n_embd, feat_size, &mut h_raw);

            // Run 8 transformer layers in raw Rust
            let new_kv_len = kv_len + 1;
            for li in 0..cfg.n_layer {
                let blk = &self.raw_blocks[li];
                let (ref mut k_heads, ref mut v_heads) = kv_raw[li];

                // Attention sublayer
                h_tmp.copy_from_slice(&h_raw);
                rms_norm_raw(&mut h_tmp, &blk.rms1, 1e-5);
                raw_gemv(&h_tmp, &blk.qkv, 3 * n_embd, n_embd, &mut qkv_buf);

                // RoPE on Q (qkv_buf[0..n_embd]) and K (qkv_buf[n_embd..2*n_embd])
                rope_single_inplace(&mut qkv_buf[..n_embd],
                                    rope_offset, &self.rope_cos_raw, &self.rope_sin_raw,
                                    n_head, head_dim);
                rope_single_inplace(&mut qkv_buf[n_embd..2 * n_embd],
                                    rope_offset, &self.rope_cos_raw, &self.rope_sin_raw,
                                    n_head, head_dim);

                // Append this token's K and V into per-head buffers
                for hi in 0..n_head {
                    k_heads[hi].extend_from_slice(
                        &qkv_buf[n_embd + hi * head_dim..n_embd + (hi + 1) * head_dim]);
                    v_heads[hi].extend_from_slice(
                        &qkv_buf[2 * n_embd + hi * head_dim..2 * n_embd + (hi + 1) * head_dim]);
                }

                mha_decode_raw(&qkv_buf[..n_embd], k_heads, v_heads,
                               n_head, head_dim, new_kv_len,
                               &mut scores_scratch, &mut attn_out);

                raw_gemv(&attn_out, &blk.c, n_embd, n_embd, &mut proj_buf);
                for i in 0..n_embd { h_raw[i] += proj_buf[i]; }

                // FFN sublayer
                h_tmp.copy_from_slice(&h_raw);
                rms_norm_raw(&mut h_tmp, &blk.rms2, 1e-5);
                silu_mlp_raw(&h_tmp, &blk.fc1, &blk.fc2, &blk.proj,
                             n_embd, mlp_hidden,
                             &mut mlp_gate, &mut mlp_up, &mut mlp_out);
                for i in 0..n_embd { h_raw[i] += mlp_out[i]; }
            }

            kv_len = new_kv_len;
            rope_offset += 1;
        }

        Ok(preds)
    }

    // -----------------------------------------------------------------------
    // Prefill (Candle path — runs once per window)
    // -----------------------------------------------------------------------

    fn prefill_block(
        &self,
        hidden: &Tensor,
        blk: &TransformerBlock,
        seq_len: usize,
        rope_start: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let res = hidden;
        let h = zsfm_nn::rms_norm(hidden, Some(&blk.rms1_w), 1e-5)?;
        let (attn_out, k, v) = self.prefill_attn(&h, blk, seq_len, rope_start)?;
        let h = (attn_out + res)?;

        let res2 = h.clone();
        let h2 = zsfm_nn::rms_norm(&h, Some(&blk.rms2_w), 1e-5)?;
        let h2 = silu_mlp(&h2, &blk.fc1_w, &blk.fc2_w, &blk.proj_w)?;
        Ok(((h2 + res2)?, k, v))
    }

    fn prefill_attn(
        &self,
        hidden: &Tensor,
        blk: &TransformerBlock,
        seq_len: usize,
        rope_start: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let n_head   = cfg.n_head;
        let head_dim = cfg.n_embd_per_head;
        let n_embd   = cfg.n_embd;

        let qkv = zsfm_nn::linear_nobias(hidden, &blk.qkv_w)?;
        let q   = qkv.narrow(1, 0, n_embd)?;
        let k   = qkv.narrow(1, n_embd, n_embd)?;
        let v   = qkv.narrow(1, 2 * n_embd, n_embd)?;

        let q = q.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let k = k.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        let q = apply_rope(&q, rope_start, seq_len, &self.rope_cos, &self.rope_sin)?;
        let k = apply_rope(&k, rope_start, seq_len, &self.rope_cos, &self.rope_sin)?;

        let scale = (head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.permute((0, 2, 1))?)? / scale)?;
        let attn_weights = self.apply_causal_mask_ll(attn_weights, seq_len)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let out = attn.matmul(&v)?;
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((seq_len, n_embd))?;
        let out = zsfm_nn::linear_nobias(&out, &blk.c_w)?;

        Ok((out, k, v))
    }

    fn apply_causal_mask_ll(&self, attn: Tensor, seq_len: usize) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.lock().unwrap();
        if !cache.contains_key(&seq_len) {
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(mask_data, (1usize, seq_len, seq_len), &self.device)?;
            cache.insert(seq_len, mask);
        }
        Ok(attn.broadcast_add(&cache[&seq_len])?)
    }
}

// ---------------------------------------------------------------------------
// Feature construction
// ---------------------------------------------------------------------------

fn build_feature_matrix(
    hist: &[f32],
    start: usize,
    seq_len: usize,
    cfg: &LagLlamaConfig,
) -> Vec<f32> {
    let mut feat = vec![0.0f32; seq_len * cfg.feature_size];
    for t in 0..seq_len {
        let abs_t = start + t;
        for (li, &lag) in cfg.lags_seq.iter().enumerate() {
            let src = abs_t as isize - lag as isize;
            if src >= 0 && (src as usize) < hist.len() {
                feat[t * cfg.feature_size + li] = hist[src as usize];
            }
        }
    }
    feat
}

fn build_one_feature(hist: &[f32], abs_t: usize, cfg: &LagLlamaConfig) -> Vec<f32> {
    let mut feat = vec![0.0f32; cfg.feature_size];
    for (li, &lag) in cfg.lags_seq.iter().enumerate() {
        let src = abs_t as isize - lag as isize;
        if src >= 0 && (src as usize) < hist.len() {
            feat[li] = hist[src as usize];
        }
    }
    feat
}

// ---------------------------------------------------------------------------
// Candle ops (used by prefill path)
// ---------------------------------------------------------------------------

/// `silu(fc1(x)) * fc2(x)`, then `proj`. Same shape as [`zsfm_nn::swiglu_ffn`]'s
/// `silu(A(x)) * C(x)` then `B(h)` — here `B` (final proj) is the 3rd param, not the
/// 2nd, so the last two args are passed to it swapped.
fn silu_mlp(x: &Tensor, fc1_w: &Tensor, fc2_w: &Tensor, proj_w: &Tensor) -> Result<Tensor> {
    zsfm_nn::swiglu_ffn(x, fc1_w, proj_w, fc2_w)
}

fn apply_rope(
    x: &Tensor,
    offset: usize,
    seq_len: usize,
    cos_table: &Tensor,
    sin_table: &Tensor,
) -> Result<Tensor> {
    let half = cos_table.dim(1)?;
    let cos_t = cos_table.narrow(0, offset, seq_len)?.unsqueeze(0)?;
    let sin_t = sin_table.narrow(0, offset, seq_len)?.unsqueeze(0)?;

    let x1 = x.narrow(D::Minus1, 0, half)?.contiguous()?;
    let x2 = x.narrow(D::Minus1, half, half)?.contiguous()?;
    let rot1 = (x1.broadcast_mul(&cos_t)? - x2.broadcast_mul(&sin_t)?)?;
    let rot2 = (x1.broadcast_mul(&sin_t)? + x2.broadcast_mul(&cos_t)?)?;
    Ok(Tensor::cat(&[&rot1, &rot2], D::Minus1)?.contiguous()?)
}

// ---------------------------------------------------------------------------
// Raw ops (used by decode path — zero Candle overhead)
// ---------------------------------------------------------------------------

simd_runtime_generate!(
    fn simd_sq_sum(row: &[f32]) -> f32 {
        let mut r = &row[..];
        let mut acc = S::Vf32::zeroes();
        while r.len() >= S::Vf32::WIDTH {
            let v = S::Vf32::load_from_slice(r);
            acc = v.mul_add(v, acc);
            r = &r[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for &x in r { sum += x * x; }
        sum
    }
);

simd_runtime_generate!(
    fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut aa = &a[..];
        let mut bb = &b[..];
        let mut acc = S::Vf32::zeroes();
        while aa.len() >= S::Vf32::WIDTH {
            let va = S::Vf32::load_from_slice(aa);
            let vb = S::Vf32::load_from_slice(bb);
            acc = va.mul_add(vb, acc);
            aa = &aa[S::Vf32::WIDTH..];
            bb = &bb[S::Vf32::WIDTH..];
        }
        let mut sum = acc.horizontal_add();
        for (&x, &y) in aa.iter().zip(bb.iter()) { sum += x * y; }
        sum
    }
);

// Schraudolph fast exp: ~0.2% relative error — sufficient for softmax and SiLU.
// ~3–5× faster than libm exp() by bypassing PLT dispatch and IEEE corner-case handling.
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    let x = x.max(-87.3365_f32); // clamp underflow to 0.0 output
    f32::from_bits(((x * 12102203.0_f32) as i32 + 1064866805_i32) as u32)
}

#[inline(always)]
fn raw_dot(a: &[f32], b: &[f32]) -> f32 {
    simd_dot(a, b)
}

fn rms_norm_raw(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len() as f32;
    let rms = (simd_sq_sum(x) / n + eps).sqrt();
    for i in 0..x.len() {
        x[i] = x[i] / rms * w[i];
    }
}

fn raw_gemv(x: &[f32], w: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    for i in 0..n_out {
        out[i] = raw_dot(x, &w[i * n_in..(i + 1) * n_in]);
    }
}

fn raw_gemv_bias(x: &[f32], w: &[f32], b: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
    for i in 0..n_out {
        out[i] = raw_dot(x, &w[i * n_in..(i + 1) * n_in]) + b[i];
    }
}

// Apply RoPE in-place to a flat [n_head, head_dim] buffer for a single token at `pos`.
fn rope_single_inplace(
    qk: &mut [f32],
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    n_head: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    let cos_row = &cos[pos * half..(pos + 1) * half];
    let sin_row = &sin[pos * half..(pos + 1) * half];
    for h in 0..n_head {
        let base = h * head_dim;
        for i in 0..half {
            let x1 = qk[base + i];
            let x2 = qk[base + half + i];
            qk[base + i]        = x1 * cos_row[i] - x2 * sin_row[i];
            qk[base + half + i] = x1 * sin_row[i] + x2 * cos_row[i];
        }
    }
}

// Single-query multi-head attention against per-head KV buffers.
// q:             [n_head * head_dim] — Q for the new token
// k_heads[h]:    [kv_len * head_dim] — all K tokens for head h
// scores_scratch: [n_head * kv_len_max] — temporary scores (stride = kv_len)
// out:           [n_head * head_dim]
fn mha_decode_raw(
    q: &[f32],
    k_heads: &[Vec<f32>],
    v_heads: &[Vec<f32>],
    n_head: usize,
    head_dim: usize,
    kv_len: usize,
    scores_scratch: &mut [f32],
    out: &mut [f32],
) {
    let scale_inv = 1.0 / (head_dim as f32).sqrt();
    for h in 0..n_head {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let k_h = &k_heads[h];
        let v_h = &v_heads[h];
        let sc  = &mut scores_scratch[h * kv_len..(h + 1) * kv_len];

        // Inline 16-element dot without function-pointer dispatch (LLVM auto-vecs to fmla.4s).
        let k_ptr = k_h.as_ptr();
        for j in 0..kv_len {
            let kj = unsafe { std::slice::from_raw_parts(k_ptr.add(j * head_dim), head_dim) };
            let mut dot = 0.0f32;
            for d in 0..head_dim { dot += q_h[d] * kj[d]; }
            sc[j] = dot * scale_inv;
        }

        // Numerically stable softmax — one division, rest multiply
        let max_s = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in sc.iter_mut() { *s = fast_exp_f32(*s - max_s); sum += *s; }
        let inv_sum = 1.0 / sum;
        for s in sc.iter_mut() { *s *= inv_sum; }

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.iter_mut().for_each(|v| *v = 0.0);
        // Safety: v_h.len() == kv_len * head_dim (invariant: prefill extraction +
        // extend_from_slice(head_dim) per step). Avoids bounds-checked slice per j.
        let v_ptr = v_h.as_ptr();
        for j in 0..kv_len {
            let sc_j = sc[j];
            let v_j = unsafe { std::slice::from_raw_parts(v_ptr.add(j * head_dim), head_dim) };
            for d in 0..head_dim {
                out_h[d] += sc_j * v_j[d];
            }
        }
    }
}

// SiLU-gated MLP: out = proj(silu(fc1(x)) * fc2(x)).
// gate and up are scratch buffers of length mlp_hidden.
fn silu_mlp_raw(
    x:         &[f32],
    fc1:       &[f32],
    fc2:       &[f32],
    proj:      &[f32],
    n_embd:    usize,
    mlp_hidden: usize,
    gate:      &mut [f32],
    up:        &mut [f32],
    out:       &mut [f32],
) {
    for i in 0..mlp_hidden {
        let v = raw_dot(x, &fc1[i * n_embd..(i + 1) * n_embd]);
        gate[i] = v / (1.0 + fast_exp_f32(-v)); // silu
    }
    for i in 0..mlp_hidden {
        up[i] = raw_dot(x, &fc2[i * n_embd..(i + 1) * n_embd]);
    }
    for j in 0..mlp_hidden { gate[j] *= up[j]; }  // gate now holds h = silu(fc1) * fc2
    raw_gemv(gate, proj, n_embd, mlp_hidden, out);
}

// Extract Candle KV caches ([n_head, seq_len, head_dim]) into per-head Vecs.
// Pre-allocates capacity for the full decode horizon to avoid realloc.
fn extract_kv_caches_raw(
    kv_caches: &[(Tensor, Tensor)],
    n_head:    usize,
    head_dim:  usize,
    horizon:   usize,
) -> Result<Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
    let mut result = Vec::with_capacity(kv_caches.len());
    for (k_t, v_t) in kv_caches {
        let k_flat = k_t.flatten_all()?.to_vec1::<f32>()?;
        let v_flat = v_t.flatten_all()?.to_vec1::<f32>()?;
        let tokens_per_head = k_flat.len() / n_head; // seq_len * head_dim
        let cap = tokens_per_head + horizon * head_dim;
        let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(n_head);
        let mut v_heads: Vec<Vec<f32>> = Vec::with_capacity(n_head);
        for h in 0..n_head {
            let start = h * tokens_per_head;
            let end   = start + tokens_per_head;
            let mut kh = Vec::with_capacity(cap);
            kh.extend_from_slice(&k_flat[start..end]);
            let mut vh = Vec::with_capacity(cap);
            vh.extend_from_slice(&v_flat[start..end]);
            k_heads.push(kh);
            v_heads.push(vh);
        }
        result.push((k_heads, v_heads));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Robust scaler
// ---------------------------------------------------------------------------

fn robust_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    if n == 0 { return (0.0, 1.0); }
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    let mut devs: Vec<f32> = sorted.iter().map(|&v| (v - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n % 2 == 0 {
        (devs[n / 2 - 1] + devs[n / 2]) / 2.0
    } else {
        devs[n / 2]
    };
    (median, mad)
}

// ---------------------------------------------------------------------------
// zsfm-core::Forecaster
// ---------------------------------------------------------------------------

impl zsfm_core::Forecaster for LagLlamaModel {
    type Config = LagLlamaConfig;

    fn load(gguf_path: &Path, config: LagLlamaConfig) -> Result<Self> {
        LagLlamaModel::load(gguf_path, config)
    }

    /// Lag-Llama is univariate-only and point-forecast-only; `mask` is unused.
    fn forecast(
        &self,
        context: &[Vec<f32>],
        _mask: &[Vec<bool>],
        horizon: usize,
    ) -> Result<zsfm_core::QuantileMatrix> {
        anyhow::ensure!(context.len() == 1, "LagLlamaModel only supports univariate forecasting (1 variate)");
        let point = LagLlamaModel::forecast(self, &context[0], horizon)?;
        Ok(vec![vec![point]])
    }
}
