//! Lag-Llama inference on the Burn backend (Flex CPU) for the prefill
//! stage, with the original verbatim pure-Rust decode loop.
//!
//! Bit-exact port strategy: Phase 1 (prefill over the full context) runs
//! the same math as `super` on Burn tensors; Phase 2 (autoregressive
//! decode) reuses the exact host kernels from `super` (`rms_norm_raw`,
//! `raw_gemv`, `rope_single_inplace`, `mha_decode_raw`, `silu_mlp_raw`)
//! copied verbatim, since they never touched candle. Weight loading goes
//! through the existing candle GGUF reader and converts each F32 tensor to
//! Burn `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use simdeez::prelude::*;
use simdeez::prelude::*;
use zsfm_burn::linear::{burn_linear_nd, burn_linear_nobias};
use zsfm_burn::norm::burn_rms_norm_nd;

use crate::config::LagLlamaConfig;

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

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnTransformerBlock {
    rms1_w: Tensor<1>,
    rms2_w: Tensor<1>,
    qkv_w: Tensor<2>,
    c_w: Tensor<2>,
    fc1_w: Tensor<2>,
    fc2_w: Tensor<2>,
    proj_w: Tensor<2>,
}

struct RawBlockWeights {
    rms1: Vec<f32>,
    rms2: Vec<f32>,
    qkv: Vec<f32>,
    c: Vec<f32>,
    fc1: Vec<f32>,
    fc2: Vec<f32>,
    proj: Vec<f32>,
}

pub struct BurnLagLlamaModel {
    config: LagLlamaConfig,
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    wte_w: Tensor<2>,
    wte_b: Tensor<1>,
    blocks: Vec<BurnTransformerBlock>,
    raw_blocks: Vec<RawBlockWeights>,
    rope_cos_raw: Vec<f32>,
    rope_sin_raw: Vec<f32>,
    norm_f_raw: Vec<f32>,
    wte_w_raw: Vec<f32>,
    wte_b_raw: Vec<f32>,
    mu_w_raw: Vec<f32>,
    mu_b_raw: Vec<f32>,
}

impl BurnLagLlamaModel {
    pub fn load(gguf_path: &Path, config: LagLlamaConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let wte_w = load_t(&content, &mut reader, "enc.wte.weight")?;
        let wte_b = {
            let t = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                "enc.wte.bias",
                &CDevice::Cpu,
                DType::F32,
            )?;
            burn1(&t)?
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for n in 0..config.n_layer {
            let p = |s: &str| format!("blk.{n}.{s}");
            let qw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_q.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let kvw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_kv.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let qd: Vec<f32> = qw.flatten_all()?.to_vec1()?;
            let kd: Vec<f32> = kvw.flatten_all()?.to_vec1()?;
            let dev = device();
            let d_model = config.n_embd;
            let mut cat = Vec::with_capacity(3 * d_model * d_model);
            cat.extend_from_slice(&qd);
            cat.extend_from_slice(&kd);
            // kv holds K and V stacked: check total size.
            let qkv_w = Tensor::<2>::from_data(TensorData::new(cat, [3 * d_model, d_model]), &dev);
            blocks.push(BurnTransformerBlock {
                rms1_w: {
                    let t = zsfm_nn::load_tensor(
                        &content,
                        &mut reader,
                        &p("rms1.weight"),
                        &CDevice::Cpu,
                        DType::F32,
                    )?;
                    burn1(&t)?
                },
                rms2_w: {
                    let t = zsfm_nn::load_tensor(
                        &content,
                        &mut reader,
                        &p("rms2.weight"),
                        &CDevice::Cpu,
                        DType::F32,
                    )?;
                    burn1(&t)?
                },
                qkv_w,
                c_w: load_t(&content, &mut reader, &p("attn_c.weight"))?,
                fc1_w: load_t(&content, &mut reader, &p("mlp_fc1.weight"))?,
                fc2_w: load_t(&content, &mut reader, &p("mlp_fc2.weight"))?,
                proj_w: load_t(&content, &mut reader, &p("mlp_proj.weight"))?,
            });
        }

        // norm_f is 1D in the checkpoint; load as vector.
        let norm_f_t = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "norm_f.weight",
            &CDevice::Cpu,
            DType::F32,
        )?;
        let norm_f_raw: Vec<f32> = norm_f_t.flatten_all()?.to_vec1()?;
        let mu_w_t = load_t(&content, &mut reader, "head.mu.weight")?;
        let mu_b_t = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "head.mu.bias",
            &CDevice::Cpu,
            DType::F32,
        )?;
        let mu_w_raw: Vec<f32> = to_host(&mu_w_t)?;
        let mu_b_raw: Vec<f32> = mu_b_t.flatten_all()?.to_vec1()?;
        let wte_w_raw: Vec<f32> = to_host(&wte_w)?;
        let wte_b_raw: Vec<f32> = {
            let t = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                "enc.wte.bias",
                &CDevice::Cpu,
                DType::F32,
            )?;
            t.flatten_all()?.to_vec1()?
        };

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
        let dev = device();
        let rope_cos =
            Tensor::<2>::from_data(TensorData::new(cos_vals.clone(), [max_pos, half]), &dev);
        let rope_sin =
            Tensor::<2>::from_data(TensorData::new(sin_vals.clone(), [max_pos, half]), &dev);

        let mut raw_blocks = Vec::with_capacity(config.n_layer);
        for blk in &blocks {
            raw_blocks.push(RawBlockWeights {
                rms1: to_host(&blk.rms1_w)?,
                rms2: to_host(&blk.rms2_w)?,
                qkv: to_host(&blk.qkv_w)?,
                c: to_host(&blk.c_w)?,
                fc1: to_host(&blk.fc1_w)?,
                fc2: to_host(&blk.fc2_w)?,
                proj: to_host(&blk.proj_w)?,
            });
        }

        Ok(Self {
            config,
            rope_cos,
            rope_sin,
            wte_w,
            wte_b: burn1(&zsfm_nn::load_tensor(
                &content,
                &mut reader,
                "enc.wte.bias",
                &CDevice::Cpu,
                DType::F32,
            )?)?,
            blocks,
            raw_blocks,
            rope_cos_raw: cos_vals,
            rope_sin_raw: sin_vals,
            norm_f_raw,
            wte_w_raw,
            wte_b_raw,
            mu_w_raw,
            mu_b_raw,
        })
    }

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let max_lag = *cfg.lags_seq.iter().max().unwrap_or(&0);
        let (loc, scale) = robust_stats(context);
        let scale = scale.max(1e-8);
        let mut hist: Vec<f32> = vec![0.0; max_lag + 1];
        for &v in context {
            hist.push((v - loc) / scale);
        }
        let ctx_buf_end = hist.len();
        let ctx_buf_start = ctx_buf_end.saturating_sub(cfg.max_context_length);
        let seq_len = ctx_buf_end - ctx_buf_start;

        // --- Phase 1: Burn prefill (full sequence, once) ---
        let feat_ctx = build_feature_matrix(&hist, ctx_buf_start, seq_len, cfg);
        let dev = device();
        let x =
            Tensor::<2>::from_data(TensorData::new(feat_ctx, [seq_len, cfg.feature_size]), &dev);
        let mut h = burn_linear_nd(x, self.wte_w.clone(), Some(self.wte_b.clone()));
        let mut kv_caches: Vec<(Tensor<3>, Tensor<3>)> = Vec::with_capacity(cfg.n_layer);
        for blk in &self.blocks {
            let (h_out, k, v) = self.prefill_block(&h, blk, seq_len, ctx_buf_start)?;
            h = h_out;
            kv_caches.push((k, v));
        }
        let h_flat: Vec<f32> = to_host(&h)?;
        let n_embd = cfg.n_embd;
        let mut h_raw: Vec<f32> = h_flat[(seq_len - 1) * n_embd..seq_len * n_embd].to_vec();

        let n_head = cfg.n_head;
        let head_dim = cfg.n_embd_per_head;
        let mlp_hidden = cfg.mlp_hidden;
        let feat_size = cfg.feature_size;
        let mut kv_raw = extract_kv_caches_raw(&kv_caches, n_head, head_dim, horizon)?;
        let mut kv_len = seq_len;

        // --- Phase 2: verbatim pure-Rust decode loop ---
        let max_kv_len = seq_len + horizon;
        let mut h_tmp = vec![0.0f32; n_embd];
        let mut qkv_buf = vec![0.0f32; 3 * n_embd];
        let mut proj_buf = vec![0.0f32; n_embd];
        let mut mlp_gate = vec![0.0f32; mlp_hidden];
        let mut mlp_up = vec![0.0f32; mlp_hidden];
        let mut mlp_out = vec![0.0f32; n_embd];
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores_scratch = vec![0.0f32; n_head * max_kv_len];
        let mut preds = Vec::with_capacity(horizon);
        let mut rope_offset = ctx_buf_start + seq_len;

        for step in 0..horizon {
            h_tmp.copy_from_slice(&h_raw);
            rms_norm_raw(&mut h_tmp, &self.norm_f_raw, 1e-5);
            let pred_scaled = raw_dot(&h_tmp, &self.mu_w_raw) + self.mu_b_raw[0];
            preds.push(pred_scaled * scale + loc);
            if step == horizon - 1 {
                break;
            }
            hist.push(pred_scaled);
            let abs_t = hist.len() - 1;
            let feat_one = build_one_feature(&hist, abs_t, cfg);
            raw_gemv_bias(
                &feat_one,
                &self.wte_w_raw,
                &self.wte_b_raw,
                n_embd,
                feat_size,
                &mut h_raw,
            );
            let new_kv_len = kv_len + 1;
            for li in 0..cfg.n_layer {
                let blk = &self.raw_blocks[li];
                let (ref mut k_heads, ref mut v_heads) = kv_raw[li];
                h_tmp.copy_from_slice(&h_raw);
                rms_norm_raw(&mut h_tmp, &blk.rms1, 1e-5);
                raw_gemv(&h_tmp, &blk.qkv, 3 * n_embd, n_embd, &mut qkv_buf);
                rope_single_inplace(
                    &mut qkv_buf[..n_embd],
                    rope_offset,
                    &self.rope_cos_raw,
                    &self.rope_sin_raw,
                    n_head,
                    head_dim,
                );
                rope_single_inplace(
                    &mut qkv_buf[n_embd..2 * n_embd],
                    rope_offset,
                    &self.rope_cos_raw,
                    &self.rope_sin_raw,
                    n_head,
                    head_dim,
                );
                for hi in 0..n_head {
                    k_heads[hi].extend_from_slice(
                        &qkv_buf[n_embd + hi * head_dim..n_embd + (hi + 1) * head_dim],
                    );
                    v_heads[hi].extend_from_slice(
                        &qkv_buf[2 * n_embd + hi * head_dim..2 * n_embd + (hi + 1) * head_dim],
                    );
                }
                mha_decode_raw(
                    &qkv_buf[..n_embd],
                    k_heads,
                    v_heads,
                    n_head,
                    head_dim,
                    new_kv_len,
                    &mut scores_scratch,
                    &mut attn_out,
                );
                raw_gemv(&attn_out, &blk.c, n_embd, n_embd, &mut proj_buf);
                for i in 0..n_embd {
                    h_raw[i] += proj_buf[i];
                }
                h_tmp.copy_from_slice(&h_raw);
                rms_norm_raw(&mut h_tmp, &blk.rms2, 1e-5);
                silu_mlp_raw(
                    &h_tmp,
                    &blk.fc1,
                    &blk.fc2,
                    &blk.proj,
                    n_embd,
                    mlp_hidden,
                    &mut mlp_gate,
                    &mut mlp_up,
                    &mut mlp_out,
                );
                for i in 0..n_embd {
                    h_raw[i] += mlp_out[i];
                }
            }
            kv_len = new_kv_len;
            rope_offset += 1;
        }
        Ok(preds)
    }

    fn prefill_block(
        &self,
        hidden: &Tensor<2>,
        blk: &BurnTransformerBlock,
        seq_len: usize,
        rope_start: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let h = burn_rms_norm_nd(hidden.clone(), blk.rms1_w.clone(), 1e-5);
        let (attn_out, k, v) = self.prefill_attn(&h, blk, seq_len, rope_start)?;
        let h = hidden.clone() + attn_out;
        let h2 = burn_rms_norm_nd(h.clone(), blk.rms2_w.clone(), 1e-5);
        let h2 = silu_mlp_burn(&h2, &blk.fc1_w, &blk.fc2_w, &blk.proj_w);
        Ok((h2 + h, k, v))
    }

    fn prefill_attn(
        &self,
        hidden: &Tensor<2>,
        blk: &BurnTransformerBlock,
        seq_len: usize,
        rope_start: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let cfg = &self.config;
        let n_head = cfg.n_head;
        let head_dim = cfg.n_embd_per_head;
        let n_embd = cfg.n_embd;
        let qkv = burn_linear_nobias(hidden.clone(), blk.qkv_w.clone());
        let q: Tensor<2> = qkv.clone().narrow(1, 0, n_embd);
        let k: Tensor<2> = qkv.clone().narrow(1, n_embd, n_embd);
        let v: Tensor<2> = qkv.narrow(1, 2 * n_embd, n_embd);
        let q: Tensor<3> = q.reshape([seq_len, n_head, head_dim]).permute([1, 0, 2]);
        let k: Tensor<3> = k.reshape([seq_len, n_head, head_dim]).permute([1, 0, 2]);
        let v: Tensor<3> = v.reshape([seq_len, n_head, head_dim]).permute([1, 0, 2]);
        let q = apply_rope_3d(&q, rope_start, seq_len, &self.rope_cos, &self.rope_sin);
        let k = apply_rope_3d(&k, rope_start, seq_len, &self.rope_cos, &self.rope_sin);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let scores = q.matmul(k.clone().permute([0, 2, 1])).mul_scalar(scale);
        let mask = burn_causal_mask(seq_len);
        let scores = scores + mask;
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v.clone());
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([seq_len, n_embd]);
        let out = burn_linear_nobias(out, blk.c_w.clone());
        Ok((out, k, v))
    }
}

fn apply_rope_3d(
    x: &Tensor<3>,
    offset: usize,
    seq_len: usize,
    cos_table: &Tensor<2>,
    sin_table: &Tensor<2>,
) -> Tensor<3> {
    let half = cos_table.dims()[1];
    let cos_t = cos_table
        .clone()
        .narrow(0, offset, seq_len)
        .unsqueeze_dim::<3>(0);
    let sin_t = sin_table
        .clone()
        .narrow(0, offset, seq_len)
        .unsqueeze_dim::<3>(0);
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rot1 = x1.clone() * cos_t.clone() - x2.clone() * sin_t.clone();
    let rot2 = x1 * sin_t + x2 * cos_t;
    Tensor::cat(vec![rot1, rot2], 2)
}

fn burn_causal_mask(seq_len: usize) -> Tensor<3> {
    let dev = device();
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<2>::from_data(TensorData::new(mask_data, [seq_len, seq_len]), &dev)
        .unsqueeze_dim::<3>(0)
}

fn silu_mlp_burn(
    x: &Tensor<2>,
    fc1_w: &Tensor<2>,
    fc2_w: &Tensor<2>,
    proj_w: &Tensor<2>,
) -> Tensor<2> {
    let h = activation::silu(burn_linear_nobias(x.clone(), fc1_w.clone()));
    let h = h * burn_linear_nobias(x.clone(), fc2_w.clone());
    burn_linear_nobias(h, proj_w.clone())
}

fn extract_kv_caches_raw(
    kv_caches: &[(Tensor<3>, Tensor<3>)],
    n_head: usize,
    head_dim: usize,
    horizon: usize,
) -> Result<Vec<(Vec<Vec<f32>>, Vec<Vec<f32>>)>> {
    let mut result = Vec::with_capacity(kv_caches.len());
    for (k_t, v_t) in kv_caches {
        let k_flat: Vec<f32> = to_host(k_t)?;
        let v_flat: Vec<f32> = to_host(v_t)?;
        let seq = k_flat.len() / (n_head * head_dim);
        let cap = (seq + horizon) * head_dim;
        let mut k_heads: Vec<Vec<f32>> = Vec::with_capacity(n_head);
        let mut v_heads: Vec<Vec<f32>> = Vec::with_capacity(n_head);
        for h in 0..n_head {
            // Layout is [H, S, D] row-major: head h occupies a contiguous chunk.
            let start = h * seq * head_dim;
            let end = start + seq * head_dim;
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

fn robust_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    if n == 0 {
        return (0.0, 1.0);
    }
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
        for &x in r {
            sum += x * x;
        }
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
        for (&x, &y) in aa.iter().zip(bb.iter()) {
            sum += x * y;
        }
        sum
    }
);

#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    let x = x.max(-87.3365_f32);
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
            qk[base + i] = x1 * cos_row[i] - x2 * sin_row[i];
            qk[base + half + i] = x1 * sin_row[i] + x2 * cos_row[i];
        }
    }
}

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
        let sc = &mut scores_scratch[h * kv_len..(h + 1) * kv_len];
        let k_ptr = k_h.as_ptr();
        for j in 0..kv_len {
            let kj = unsafe { std::slice::from_raw_parts(k_ptr.add(j * head_dim), head_dim) };
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_h[d] * kj[d];
            }
            sc[j] = dot * scale_inv;
        }
        let max_s = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in sc.iter_mut() {
            *s = fast_exp_f32(*s - max_s);
            sum += *s;
        }
        let inv_sum = 1.0 / sum;
        for s in sc.iter_mut() {
            *s *= inv_sum;
        }
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.iter_mut().for_each(|v| *v = 0.0);
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

fn silu_mlp_raw(
    x: &[f32],
    fc1: &[f32],
    fc2: &[f32],
    proj: &[f32],
    n_embd: usize,
    mlp_hidden: usize,
    gate: &mut [f32],
    up: &mut [f32],
    out: &mut [f32],
) {
    for i in 0..mlp_hidden {
        let v = raw_dot(x, &fc1[i * n_embd..(i + 1) * n_embd]);
        gate[i] = v / (1.0 + fast_exp_f32(-v));
    }
    for i in 0..mlp_hidden {
        up[i] = raw_dot(x, &fc2[i * n_embd..(i + 1) * n_embd]);
    }
    for j in 0..mlp_hidden {
        gate[j] *= up[j];
    }
    raw_gemv(gate, proj, n_embd, mlp_hidden, out);
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

    fn tiny_config() -> LagLlamaConfig {
        LagLlamaConfig {
            n_layer: 1,
            n_head: 2,
            n_embd_per_head: 8,
            n_embd: 16,
            mlp_hidden: 32,
            feature_size: 4,
            n_lags: 2,
            n_time_feat: 2,
            max_context_length: 16,
            lags_seq: vec![1, 2],
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

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        let dir = std::env::temp_dir().join(format!("lagllama-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("lag-llama".into()),
        );
        let mut seed = 2700u64;
        let s = 0.1;
        put2(&mut w, "enc.wte.weight", 16, 4, &mut seed, s);
        put1(&mut w, "enc.wte.bias", 16, &mut seed);
        put2(&mut w, "blk.0.attn_q.weight", 16, 16, &mut seed, s);
        put2(&mut w, "blk.0.attn_kv.weight", 32, 16, &mut seed, s);
        put1(&mut w, "blk.0.rms1.weight", 16, &mut seed);
        put1(&mut w, "blk.0.rms2.weight", 16, &mut seed);
        put2(&mut w, "blk.0.attn_c.weight", 16, 16, &mut seed, s);
        put2(&mut w, "blk.0.mlp_fc1.weight", 32, 16, &mut seed, s);
        put2(&mut w, "blk.0.mlp_fc2.weight", 32, 16, &mut seed, s);
        put2(&mut w, "blk.0.mlp_proj.weight", 16, 32, &mut seed, s);
        put1(&mut w, "norm_f.weight", 16, &mut seed);
        put2(&mut w, "head.mu.weight", 1, 16, &mut seed, s);
        put1(&mut w, "head.mu.bias", 1, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::LagLlamaModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnLagLlamaModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, 2).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx, 2).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let err: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        println!("lag-llama synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-3, "lag-llama Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
