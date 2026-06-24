//! Lag-Llama inference engine with KV caching.
//!
//! Two-phase inference:
//! 1. Prefill: full forward pass over context tokens, collects K/V cache per layer.
//! 2. Decode:  per-step single-token Q attended to cached K/V (O(1) attention per step).

use std::collections::HashMap;
use std::sync::Mutex;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor, D};

use crate::config::LagLlamaConfig;

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

struct TransformerBlock {
    rms1_w: Tensor,
    rms2_w: Tensor,
    qkv_w: Tensor,  // fused [3*n_embd, n_embd]
    c_w: Tensor,
    fc1_w: Tensor,
    fc2_w: Tensor,
    proj_w: Tensor,
}

pub struct LagLlamaModel {
    device: Device,
    config: LagLlamaConfig,
    rope_cos: Tensor,  // [max_positions, half_head_dim] — precomputed at load
    rope_sin: Tensor,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
    wte_w: Tensor,
    wte_b: Tensor,
    blocks: Vec<TransformerBlock>,
    norm_f_w: Tensor,
    mu_w: Tensor,
    mu_b: Tensor,
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
    let qt = content
        .tensor(reader, name, device)
        .with_context(|| format!("tensor '{name}' not found in GGUF"))?;
    Ok(qt.dequantize(device)?.to_dtype(DType::F32)?)
}

impl LagLlamaModel {
    pub fn load(gguf_path: &Path, config: LagLlamaConfig) -> Result<Self> {
        let device = Device::Cpu;
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let wte_w = load_t(&content, &mut reader, "enc.wte.weight", &device)?;
        let wte_b = load_t(&content, &mut reader, "enc.wte.bias", &device)?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for n in 0..config.n_layer {
            let p = |s: &str| format!("blk.{n}.{s}");
            let q_w  = load_t(&content, &mut reader, &p("attn_q.weight"),  &device)?;
            let kv_w = load_t(&content, &mut reader, &p("attn_kv.weight"), &device)?;
            blocks.push(TransformerBlock {
                rms1_w: load_t(&content, &mut reader, &p("rms1.weight"), &device)?,
                rms2_w: load_t(&content, &mut reader, &p("rms2.weight"), &device)?,
                qkv_w:  Tensor::cat(&[&q_w, &kv_w], 0)?,
                c_w:    load_t(&content, &mut reader, &p("attn_c.weight"), &device)?,
                fc1_w:  load_t(&content, &mut reader, &p("mlp_fc1.weight"), &device)?,
                fc2_w:  load_t(&content, &mut reader, &p("mlp_fc2.weight"), &device)?,
                proj_w: load_t(&content, &mut reader, &p("mlp_proj.weight"), &device)?,
            });
        }

        let norm_f_w = load_t(&content, &mut reader, "norm_f.weight", &device)?;
        let mu_w     = load_t(&content, &mut reader, "head.mu.weight", &device)?;
        let mu_b     = load_t(&content, &mut reader, "head.mu.bias", &device)?;

        let head_dim = config.n_embd_per_head;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Precompute RoPE cos/sin tables up to max_context_length + 4096 for decode headroom
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
        let rope_cos = Tensor::from_vec(cos_vals, (max_pos, half), &device)?;
        let rope_sin = Tensor::from_vec(sin_vals, (max_pos, half), &device)?;

        Ok(Self {
            device,
            config,
            rope_cos,
            rope_sin,
            causal_mask_cache: Mutex::new(HashMap::new()),
            wte_w,
            wte_b,
            blocks,
            norm_f_w,
            mu_w,
            mu_b,
        })
    }

    // -----------------------------------------------------------------------
    // Forecasting (KV-cached)
    // -----------------------------------------------------------------------

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let max_lag = *cfg.lags_seq.iter().max().unwrap_or(&0);

        let (loc, scale) = robust_stats(context);
        let scale = scale.max(1e-8);

        // History buffer: zero-padded at front so every lag is valid.
        let mut hist: Vec<f32> = vec![0.0; max_lag + 1];
        for &v in context {
            hist.push((v - loc) / scale);
        }

        // Clamp context to max_context_length (model was trained with this limit).
        let ctx_buf_end = hist.len();                        // exclusive index in hist
        let ctx_buf_start = ctx_buf_end.saturating_sub(cfg.max_context_length);
        let seq_len = ctx_buf_end - ctx_buf_start;

        // --- Phase 1: prefill over context tokens ---
        let feat_ctx = build_feature_matrix(&hist, ctx_buf_start, seq_len, cfg);
        let x = Tensor::from_vec(feat_ctx, (seq_len, cfg.feature_size), &self.device)?;
        let mut h = linear_bias(&x, &self.wte_w, &self.wte_b)?;

        // Per-layer KV caches: Vec<(K, V)> each [n_head, seq, head_dim]
        let mut kv_caches: Vec<(Tensor, Tensor)> = Vec::with_capacity(cfg.n_layer);

        for blk in &self.blocks {
            let (h_out, k, v) = self.prefill_block(&h, blk, seq_len)?;
            h = h_out;
            kv_caches.push((k, v));
        }

        // Last token hidden state → first prediction
        let mut last_h = h.get(seq_len - 1)?;

        let mut preds = Vec::with_capacity(horizon);

        // Rope offset starts at seq_len (next token position)
        let mut rope_offset = seq_len;

        for step in 0..horizon {
            // Apply final norm and head to get prediction
            let normed = rms_norm(&last_h, &self.norm_f_w)?;
            let mu = linear_bias(&normed.unsqueeze(0)?, &self.mu_w, &self.mu_b)?;
            let pred_scaled = mu.flatten_all()?.get(0)?.to_scalar::<f32>()?;
            preds.push(pred_scaled * scale + loc);

            if step == horizon - 1 { break; }

            // Append prediction to history for next step's lag features
            hist.push(pred_scaled);

            // Build 1-token feature for next decode step
            let abs_t = hist.len() - 1; // index of the just-appended token
            let feat_one = build_one_feature(&hist, abs_t, cfg);
            let x1 = Tensor::from_vec(feat_one, (1, cfg.feature_size), &self.device)?;
            let mut h1 = linear_bias(&x1, &self.wte_w, &self.wte_b)?;

            for (li, blk) in self.blocks.iter().enumerate() {
                let (h1_out, new_k, new_v) = self.decode_block(&h1, blk, &kv_caches[li], rope_offset)?;
                h1 = h1_out;
                // new_k/new_v are the single-token post-RoPE K/V; append to cache
                let cat_k = Tensor::cat(&[&kv_caches[li].0, &new_k], 1)?;
                let cat_v = Tensor::cat(&[&kv_caches[li].1, &new_v], 1)?;
                kv_caches[li] = (cat_k, cat_v);
            }

            last_h = h1.get(0)?;
            rope_offset += 1;
        }

        Ok(preds)
    }

    // Prefill: full causal attention over seq_len tokens.
    // Returns (hidden, K, V) where K/V are [n_head, seq_len, head_dim].
    fn prefill_block(
        &self,
        hidden: &Tensor,
        blk: &TransformerBlock,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let res = hidden;
        let h = rms_norm(hidden, &blk.rms1_w)?;
        let (attn_out, k, v) = self.prefill_attn(&h, blk, seq_len)?;
        let h = (attn_out + res)?;

        let res2 = h.clone();
        let h2 = rms_norm(&h, &blk.rms2_w)?;
        let h2 = silu_mlp(&h2, &blk.fc1_w, &blk.fc2_w, &blk.proj_w)?;
        Ok(((h2 + res2)?, k, v))
    }

    fn prefill_attn(
        &self,
        hidden: &Tensor,
        blk: &TransformerBlock,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let n_head = cfg.n_head;
        let head_dim = cfg.n_embd_per_head;
        let n_embd = cfg.n_embd;

        let qkv = linear_nobias(hidden, &blk.qkv_w)?;
        let q  = qkv.narrow(1, 0, n_embd)?;
        let k  = qkv.narrow(1, n_embd, n_embd)?;
        let v  = qkv.narrow(1, 2 * n_embd, n_embd)?;

        let q = q.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let k = k.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((seq_len, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        let q = apply_rope(&q, 0, seq_len, &self.rope_cos, &self.rope_sin)?;
        let k = apply_rope(&k, 0, seq_len, &self.rope_cos, &self.rope_sin)?;

        let scale = (head_dim as f64).sqrt();
        let attn = q.matmul(&k.permute((0, 2, 1))?)?;
        let attn = (attn / scale)?;
        let attn = self.apply_causal_mask_ll(attn, seq_len)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((seq_len, n_embd))?;
        let out = linear_nobias(&out, &blk.c_w)?;

        Ok((out, k, v))
    }

    // Decode: single new token attended to cached K/V.
    // Returns (hidden, new_K, new_V) where new_K/V are [n_head, 1, head_dim].
    fn decode_block(
        &self,
        hidden: &Tensor,         // [1, n_embd]
        blk: &TransformerBlock,
        cache: &(Tensor, Tensor), // ([n_head, cached, head_dim], same)
        rope_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let res = hidden;
        let h = rms_norm(hidden, &blk.rms1_w)?;
        let (attn_out, new_k, new_v) = self.decode_attn(&h, blk, cache, rope_offset)?;
        let h = (attn_out + res)?;

        let res2 = h.clone();
        let h2 = rms_norm(&h, &blk.rms2_w)?;
        let h2 = silu_mlp(&h2, &blk.fc1_w, &blk.fc2_w, &blk.proj_w)?;
        Ok(((h2 + res2)?, new_k, new_v))
    }

    fn decode_attn(
        &self,
        hidden: &Tensor,          // [1, n_embd]
        blk: &TransformerBlock,
        cache: &(Tensor, Tensor),  // K/V caches [n_head, cached_len, head_dim]
        rope_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let n_head = cfg.n_head;
        let head_dim = cfg.n_embd_per_head;
        let n_embd = cfg.n_embd;

        let qkv = linear_nobias(hidden, &blk.qkv_w)?;  // [1, 3*n_embd]
        let q  = qkv.narrow(1, 0, n_embd)?;
        let k  = qkv.narrow(1, n_embd, n_embd)?;
        let v  = qkv.narrow(1, 2 * n_embd, n_embd)?;

        // Reshape to [n_head, 1, head_dim]
        let q = q.reshape((1, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let k = k.reshape((1, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;
        let v = v.reshape((1, n_head, head_dim))?.permute((1, 0, 2))?.contiguous()?;

        // Apply RoPE with offset
        let q = apply_rope(&q, rope_offset, 1, &self.rope_cos, &self.rope_sin)?;
        let k_rope = apply_rope(&k, rope_offset, 1, &self.rope_cos, &self.rope_sin)?;

        // Concatenate with cache: [n_head, cached+1, head_dim]
        let k_full = Tensor::cat(&[&cache.0, &k_rope], 1)?;
        let v_full = Tensor::cat(&[&cache.1, &v], 1)?;

        // Attention: [n_head, 1, head_dim] × [n_head, head_dim, total] → [n_head, 1, total]
        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(&k_full.permute((0, 2, 1))?)?;
        let scores = (scores / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted: [n_head, 1, total] × [n_head, total, head_dim] → [n_head, 1, head_dim]
        let out = attn.matmul(&v_full)?;
        let out = out.permute((1, 0, 2))?.contiguous()?.reshape((1, n_embd))?;
        let out = linear_nobias(&out, &blk.c_w)?;

        // Return RoPE-encoded k so the caller stores post-RoPE keys in the cache
        Ok((out, k_rope, v))
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
// Ops
// ---------------------------------------------------------------------------

fn linear_nobias(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?)
}

fn linear_bias(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(x.matmul(&w.t()?)?.broadcast_add(b)?)
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let rms = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (rms + 1e-5_f64)?.sqrt()?;
    let x = x.broadcast_div(&rms)?;
    Ok(x.broadcast_mul(weight)?)
}

fn silu_mlp(x: &Tensor, fc1_w: &Tensor, fc2_w: &Tensor, proj_w: &Tensor) -> Result<Tensor> {
    let gate = linear_nobias(x, fc1_w)?.silu()?;
    let val  = linear_nobias(x, fc2_w)?;
    let h = (gate * val)?;
    linear_nobias(&h, proj_w)
}

/// RoPE: x is [n_head, seq, head_dim]; slices precomputed cos/sin from table.
fn apply_rope(
    x: &Tensor,
    offset: usize,
    seq_len: usize,
    cos_table: &Tensor,  // [max_pos, half]
    sin_table: &Tensor,
) -> Result<Tensor> {
    let half = cos_table.dim(1)?;
    let cos_t = cos_table.narrow(0, offset, seq_len)?.unsqueeze(0)?;  // [1, seq, half]
    let sin_t = sin_table.narrow(0, offset, seq_len)?.unsqueeze(0)?;

    let x1 = x.narrow(D::Minus1, 0, half)?.contiguous()?;
    let x2 = x.narrow(D::Minus1, half, half)?.contiguous()?;
    let rot1 = (x1.broadcast_mul(&cos_t)? - x2.broadcast_mul(&sin_t)?)?;
    let rot2 = (x1.broadcast_mul(&sin_t)? + x2.broadcast_mul(&cos_t)?)?;
    Ok(Tensor::cat(&[&rot1, &rot2], D::Minus1)?.contiguous()?)
}

impl LagLlamaModel {
    fn apply_causal_mask_ll(&self, attn: Tensor, seq_len: usize) -> Result<Tensor> {
        let mut cache = self.causal_mask_cache.lock().unwrap();
        if !cache.contains_key(&seq_len) {
            // [1, seq_len, seq_len] — broadcast across heads
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
