//! Moirai-2.0 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): packed scaler, patch tokenizer,
//! prefill + KV-cached decode with partial RoPE, QK-norm attention,
//! same-variate bias, SwiGLU FFN, quantile output head. Weight loading goes
//! through the existing candle GGUF reader and converts each F32 tensor to
//! Burn `TensorData`, so there is exactly one GGUF parser.
//!
//! Partial RoPE runs on host data via `zsfm_burn::rope::apply_interleaved_rope`
//! (Flex CPU: the roundtrip is cheap relative to attention).

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::{burn_linear_bias, burn_linear_nobias};
use zsfm_burn::norm::burn_rms_norm_nd;
use zsfm_burn::rope::apply_interleaved_rope;

use crate::config::Moirai2Config;

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

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnResidualBlockW {
    hidden_w: Tensor<2>,
    hidden_b: Tensor<1>,
    output_w: Tensor<2>,
    output_b: Tensor<1>,
    residual_w: Tensor<2>,
    residual_b: Tensor<1>,
}

struct BurnEncoderBlock {
    norm1_w: Tensor<1>,
    norm2_w: Tensor<1>,
    attn_qkv_w: Tensor<2>,
    attn_o_w: Tensor<2>,
    attn_qn_w: Tensor<1>,
    attn_kn_w: Tensor<1>,
    attn_vbias_t: Tensor<3>,
    ffn_fc1_w: Tensor<2>,
    ffn_fc2_w: Tensor<2>,
    ffn_gate_w: Tensor<2>,
}

pub struct BurnMoirai2Model {
    config: Moirai2Config,
    in_proj: BurnResidualBlockW,
    blocks: Vec<BurnEncoderBlock>,
    norm_f_w: Tensor<1>,
    out_proj: BurnResidualBlockW,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnResidualBlockW> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(BurnResidualBlockW {
        hidden_w: load_t(content, reader, &p("hidden.weight"))?,
        hidden_b: load_v(content, reader, &p("hidden.bias"))?,
        output_w: load_t(content, reader, &p("output.weight"))?,
        output_b: load_v(content, reader, &p("output.bias"))?,
        residual_w: load_t(content, reader, &p("residual.weight"))?,
        residual_b: load_v(content, reader, &p("residual.bias"))?,
    })
}

fn residual_block_fwd(x: Tensor<2>, w: &BurnResidualBlockW) -> Tensor<2> {
    let hidden = activation::silu(burn_linear_bias(
        x.clone(),
        w.hidden_w.clone(),
        w.hidden_b.clone(),
    ));
    let output = burn_linear_bias(hidden, w.output_w.clone(), w.output_b.clone());
    let residual = burn_linear_bias(x, w.residual_w.clone(), w.residual_b.clone());
    output + residual
}

fn qk_norm_heads(
    x: Tensor<3>,
    weight: Tensor<1>,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor<3> {
    burn_rms_norm_nd(x.reshape([seq_len * n_heads, head_dim]), weight, 1e-6)
        .reshape([seq_len, n_heads, head_dim])
}

fn make_causal_mask_t(seq_len: usize) -> Tensor<3> {
    let dev = device();
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::<3>::from_data(TensorData::new(mask, [1, seq_len, seq_len]), &dev)
}

fn add_decode_causal_mask(scores: Tensor<3>, new_len: usize, cached_len: usize) -> Tensor<3> {
    let dev = device();
    let total = cached_len + new_len;
    let mut mask = vec![0.0f32; new_len * total];
    for q_rel in 0..new_len {
        for k_abs in (cached_len + q_rel + 1)..total {
            mask[q_rel * total + k_abs] = f32::NEG_INFINITY;
        }
    }
    let mask_t = Tensor::<3>::from_data(TensorData::new(mask, [1, new_len, total]), &dev);
    scores + mask_t
}

fn swiglu_ffn(x: Tensor<2>, fc1_w: Tensor<2>, fc2_w: Tensor<2>, gate_w: Tensor<2>) -> Tensor<2> {
    let content = activation::silu(burn_linear_nobias(x.clone(), fc1_w));
    let gate = burn_linear_nobias(x, gate_w);
    burn_linear_nobias(content * gate, fc2_w)
}

impl BurnMoirai2Model {
    pub fn load(gguf_path: &Path, config: Moirai2Config) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;
        let dev = device();

        let in_proj = load_residual_block(&content, &mut reader, "in_proj")?;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for n in 0..config.n_layers {
            let p = |s: &str| format!("blk.{n}.{s}");
            let qw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_q.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let kw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_k.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let vw = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_v.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?;
            let d_model = config.d_model;
            let mut cat = Vec::with_capacity(3 * d_model * d_model);
            for t in [&qw, &kw, &vw] {
                cat.extend_from_slice(&t.flatten_all()?.to_vec1::<f32>()?);
            }
            let attn_qkv_w =
                Tensor::<2>::from_data(TensorData::new(cat, [3 * d_model, d_model]), &dev);
            let vbias_raw: Vec<f32> = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                &p("attn_vbias.weight"),
                &CDevice::Cpu,
                DType::F32,
            )?
            .flatten_all()?
            .to_vec1()?;
            let n_heads = config.n_heads;
            let same_var_bias: Vec<f32> = (0..n_heads).map(|h| vbias_raw[n_heads + h]).collect();
            let attn_vbias_t =
                Tensor::<3>::from_data(TensorData::new(same_var_bias, [n_heads, 1, 1]), &dev);
            blocks.push(BurnEncoderBlock {
                norm1_w: load_v(&content, &mut reader, &p("norm1.weight"))?,
                norm2_w: load_v(&content, &mut reader, &p("norm2.weight"))?,
                attn_qkv_w,
                attn_o_w: load_t(&content, &mut reader, &p("attn_o.weight"))?,
                attn_qn_w: load_v(&content, &mut reader, &p("attn_qn.weight"))?,
                attn_kn_w: load_v(&content, &mut reader, &p("attn_kn.weight"))?,
                attn_vbias_t,
                ffn_fc1_w: load_t(&content, &mut reader, &p("ffn_fc1.weight"))?,
                ffn_fc2_w: load_t(&content, &mut reader, &p("ffn_fc2.weight"))?,
                ffn_gate_w: load_t(&content, &mut reader, &p("ffn_gate.weight"))?,
            });
        }
        let norm_f_w = load_v(&content, &mut reader, "norm_f.weight")?;
        let out_proj = load_residual_block(&content, &mut reader, "out_proj")?;

        let half_rope = config.rope_dim / 2;
        let max_pos = config.max_ctx_tokens() + 256;
        let inv_freq: Vec<f32> = (0..half_rope)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / config.rope_dim as f32))
            .collect();
        let mut rope_cos = vec![0.0f32; max_pos * half_rope];
        let mut rope_sin = vec![0.0f32; max_pos * half_rope];
        for pos in 0..max_pos {
            for i in 0..half_rope {
                let theta = pos as f32 * inv_freq[i];
                rope_cos[pos * half_rope + i] = theta.cos();
                rope_sin[pos * half_rope + i] = theta.sin();
            }
        }
        Ok(Self {
            config,
            in_proj,
            blocks,
            norm_f_w,
            out_proj,
            rope_cos,
            rope_sin,
        })
    }

    fn apply_partial_rope(
        &self,
        x: Tensor<3>,
        time_ids: &[usize],
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor<3>> {
        let dims = x.dims();
        let mut data = to_host(&x)?;
        apply_interleaved_rope(
            &mut data,
            time_ids,
            n_heads,
            dims[1],
            head_dim,
            self.config.rope_dim,
            &self.rope_cos,
            &self.rope_sin,
        )?;
        Ok(Tensor::<3>::from_data(
            TensorData::new(data, dims),
            &device(),
        ))
    }

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let ps = cfg.patch_size;
        let max_ctx_len = cfg.max_ctx_tokens() * ps;
        let ctx: &[f32] = if context.len() > max_ctx_len {
            &context[context.len() - max_ctx_len..]
        } else {
            context
        };
        let n = ctx.len() as f64;
        let loc = ctx.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = ctx.iter().map(|&v| (v as f64 - loc).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let scale = ((var + 1e-5_f64).sqrt()) as f32;

        let ctx_norm: Vec<f32> = ctx.iter().map(|&v| (v - loc as f32) / scale).collect();
        let rem = ctx_norm.len() % ps;
        let ctx_padded: Vec<f32> = if rem != 0 {
            let mut padded = vec![0.0f32; ps - rem];
            padded.extend_from_slice(&ctx_norm);
            padded
        } else {
            ctx_norm
        };
        let n_ctx = ctx_padded.len() / ps;
        let ctx_flat: Vec<f32> = (0..n_ctx)
            .flat_map(|i| {
                let mut tok = ctx_padded[i * ps..(i + 1) * ps].to_vec();
                tok.extend(vec![1.0f32; ps]);
                tok
            })
            .collect();
        let ctx_time_ids: Vec<usize> = (0..n_ctx).collect();
        let num_pt = cfg.num_predict_token;
        let num_q = cfg.num_quantiles;
        let mq = cfg.median_quantile;
        let n_future_patches = (horizon + ps - 1) / ps;

        let dev = device();
        let input_t = Tensor::<2>::from_data(TensorData::new(ctx_flat, [n_ctx, ps * 2]), &dev);
        let h_ctx = residual_block_fwd(input_t, &self.in_proj);
        let (h_enc, mut kv_cache) = self.prefill_encoder(h_ctx, &ctx_time_ids, n_ctx)?;
        let last: Tensor<2> = h_enc.narrow(0, n_ctx - 1, 1);
        let last_norm = burn_rms_norm_nd(last, self.norm_f_w.clone(), 1e-6);
        let first_pred: Vec<f32> = to_host(&residual_block_fwd(last_norm, &self.out_proj))?;
        let mut prev_patches: Vec<Vec<f32>> = (0..num_pt)
            .map(|pt| {
                let q_start = pt * num_q * ps + mq * ps;
                first_pred[q_start..q_start + ps].to_vec()
            })
            .collect();
        let mut collected_patches: Vec<Vec<f32>> = Vec::new();
        let n_take = n_future_patches.min(num_pt);
        for patch in prev_patches.iter().take(n_take) {
            collected_patches.push(patch.clone());
        }

        let mut cached_len = n_ctx;
        while collected_patches.len() < n_future_patches {
            let new_time_ids: Vec<usize> = (cached_len..cached_len + num_pt).collect();
            let new_flat: Vec<f32> = prev_patches
                .iter()
                .flat_map(|patch| {
                    let mut tok = patch.clone();
                    tok.extend(vec![0.0f32; ps]);
                    tok
                })
                .collect();
            let new_t = Tensor::<2>::from_data(TensorData::new(new_flat, [num_pt, ps * 2]), &dev);
            let h_in = residual_block_fwd(new_t, &self.in_proj);
            let h_dec = self.decode_encoder(h_in, &new_time_ids, &mut kv_cache, cached_len)?;
            let last2: Tensor<2> = h_dec.narrow(0, num_pt - 1, 1);
            let last_norm2 = burn_rms_norm_nd(last2, self.norm_f_w.clone(), 1e-6);
            let pred: Vec<f32> = to_host(&residual_block_fwd(last_norm2, &self.out_proj))?;
            let new_patches: Vec<Vec<f32>> = (0..num_pt)
                .map(|pt| {
                    let q_start = pt * num_q * ps + mq * ps;
                    pred[q_start..q_start + ps].to_vec()
                })
                .collect();
            let need = n_future_patches - collected_patches.len();
            for patch in new_patches.iter().take(need.min(num_pt)) {
                collected_patches.push(patch.clone());
            }
            cached_len += num_pt;
            prev_patches = new_patches;
        }

        Ok(collected_patches
            .iter()
            .flat_map(|p| p.iter().copied())
            .take(horizon)
            .map(|v| v * scale + loc as f32)
            .collect())
    }

    fn prefill_encoder(
        &self,
        mut h: Tensor<2>,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor<2>, Vec<(Tensor<3>, Tensor<3>)>)> {
        let mut kv_cache = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (h_out, k, v) = self.prefill_block(h, blk, time_ids, seq_len)?;
            h = h_out;
            kv_cache.push((k, v));
        }
        Ok((h, kv_cache))
    }

    fn prefill_block(
        &self,
        h: Tensor<2>,
        blk: &BurnEncoderBlock,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let h_norm = burn_rms_norm_nd(h.clone(), blk.norm1_w.clone(), 1e-6);
        let (attn_out, k, v) = self.prefill_attn(&h_norm, blk, time_ids, seq_len)?;
        let h = h + attn_out;
        let h_norm2 = burn_rms_norm_nd(h.clone(), blk.norm2_w.clone(), 1e-6);
        let ffn_out = swiglu_ffn(
            h_norm2,
            blk.ffn_fc1_w.clone(),
            blk.ffn_fc2_w.clone(),
            blk.ffn_gate_w.clone(),
        );
        Ok((h + ffn_out, k, v))
    }

    fn prefill_attn(
        &self,
        h: &Tensor<2>,
        blk: &BurnEncoderBlock,
        time_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let cfg = &self.config;
        let qkv = burn_linear_nobias(h.clone(), blk.attn_qkv_w.clone());
        let d = cfg.d_model;
        let q: Tensor<2> = qkv.clone().narrow(1, 0, d);
        let k: Tensor<2> = qkv.clone().narrow(1, d, d);
        let v: Tensor<2> = qkv.narrow(1, 2 * d, d);
        let q: Tensor<3> = q.reshape([seq_len, cfg.n_heads, cfg.head_dim]);
        let k: Tensor<3> = k.reshape([seq_len, cfg.n_heads, cfg.head_dim]);
        let q = qk_norm_heads(q, blk.attn_qn_w.clone(), seq_len, cfg.n_heads, cfg.head_dim);
        let k = qk_norm_heads(k, blk.attn_kn_w.clone(), seq_len, cfg.n_heads, cfg.head_dim);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let v: Tensor<3> = v
            .reshape([seq_len, cfg.n_heads, cfg.head_dim])
            .permute([1, 0, 2]);
        let q = self.apply_partial_rope(q, time_ids, cfg.n_heads, cfg.head_dim)?;
        let k = self.apply_partial_rope(k.clone(), time_ids, cfg.n_heads, cfg.head_dim)?;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let scores = q.matmul(k.clone().permute([0, 2, 1])).mul_scalar(scale);
        let causal = make_causal_mask_t(seq_len);
        let scores = scores + causal;
        let scores = scores + blk.attn_vbias_t.clone();
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v.clone());
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([seq_len, d]);
        Ok((burn_linear_nobias(out, blk.attn_o_w.clone()), k, v))
    }

    fn decode_encoder(
        &self,
        mut h: Tensor<2>,
        new_time_ids: &[usize],
        kv_cache: &mut Vec<(Tensor<3>, Tensor<3>)>,
        cached_len: usize,
    ) -> Result<Tensor<2>> {
        for (li, blk) in self.blocks.iter().enumerate() {
            let (h_out, k_new, v_new) =
                self.decode_block_kv(h, blk, &kv_cache[li], new_time_ids, cached_len)?;
            h = h_out;
            kv_cache[li] = (k_new, v_new);
        }
        Ok(h)
    }

    fn decode_block_kv(
        &self,
        h: Tensor<2>,
        blk: &BurnEncoderBlock,
        cache: &(Tensor<3>, Tensor<3>),
        new_time_ids: &[usize],
        cached_len: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let h_norm = burn_rms_norm_nd(h.clone(), blk.norm1_w.clone(), 1e-6);
        let (attn_out, k_new, v_new) =
            self.decode_attn_kv(&h_norm, blk, cache, new_time_ids, cached_len)?;
        let h = h + attn_out;
        let h_norm2 = burn_rms_norm_nd(h.clone(), blk.norm2_w.clone(), 1e-6);
        let ffn_out = swiglu_ffn(
            h_norm2,
            blk.ffn_fc1_w.clone(),
            blk.ffn_fc2_w.clone(),
            blk.ffn_gate_w.clone(),
        );
        Ok((h + ffn_out, k_new, v_new))
    }

    fn decode_attn_kv(
        &self,
        h: &Tensor<2>,
        blk: &BurnEncoderBlock,
        cache: &(Tensor<3>, Tensor<3>),
        new_time_ids: &[usize],
        cached_len: usize,
    ) -> Result<(Tensor<2>, Tensor<3>, Tensor<3>)> {
        let cfg = &self.config;
        let new_len = new_time_ids.len();
        let d = cfg.d_model;
        let qkv = burn_linear_nobias(h.clone(), blk.attn_qkv_w.clone());
        let q: Tensor<2> = qkv.clone().narrow(1, 0, d);
        let k: Tensor<2> = qkv.clone().narrow(1, d, d);
        let v: Tensor<2> = qkv.narrow(1, 2 * d, d);
        let q: Tensor<3> = q.reshape([new_len, cfg.n_heads, cfg.head_dim]);
        let k: Tensor<3> = k.reshape([new_len, cfg.n_heads, cfg.head_dim]);
        let q = qk_norm_heads(q, blk.attn_qn_w.clone(), new_len, cfg.n_heads, cfg.head_dim);
        let k = qk_norm_heads(k, blk.attn_kn_w.clone(), new_len, cfg.n_heads, cfg.head_dim);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let v: Tensor<3> = v
            .reshape([new_len, cfg.n_heads, cfg.head_dim])
            .permute([1, 0, 2]);
        let q = self.apply_partial_rope(q, new_time_ids, cfg.n_heads, cfg.head_dim)?;
        let k = self.apply_partial_rope(k, new_time_ids, cfg.n_heads, cfg.head_dim)?;
        let k_full = Tensor::cat(vec![cache.0.clone(), k], 1);
        let v_full = Tensor::cat(vec![cache.1.clone(), v], 1);
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let scores = q
            .matmul(k_full.clone().permute([0, 2, 1]))
            .mul_scalar(scale);
        let scores = add_decode_causal_mask(scores, new_len, cached_len);
        let scores = scores + blk.attn_vbias_t.clone();
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v_full.clone());
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([new_len, d]);
        Ok((
            burn_linear_nobias(out, blk.attn_o_w.clone()),
            k_full,
            v_full,
        ))
    }
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

    fn tiny_config() -> Moirai2Config {
        Moirai2Config {
            d_model: 16,
            n_layers: 1,
            n_heads: 2,
            head_dim: 8,
            d_ff: 32,
            patch_size: 2,
            num_predict_token: 2,
            num_quantiles: 3,
            max_seq_len: 16,
            rope_dim: 4,
            median_quantile: 1,
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

    fn put_res(
        w: &mut GGUFWriter,
        prefix: &str,
        h: usize,
        d_in: usize,
        o: usize,
        seed: &mut u64,
        s: f32,
    ) {
        put2(w, &format!("{prefix}.hidden.weight"), h, d_in, seed, s);
        put1(w, &format!("{prefix}.hidden.bias"), h, seed);
        put2(w, &format!("{prefix}.output.weight"), o, h, seed, s);
        put1(w, &format!("{prefix}.output.bias"), o, seed);
        put2(w, &format!("{prefix}.residual.weight"), o, d_in, seed, s);
        put1(w, &format!("{prefix}.residual.bias"), o, seed);
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        // d=16, heads=2, hd=8, 1 layer, ps=2, pt=2, q=3, rope=4.
        let dir = std::env::temp_dir().join(format!("moirai2-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("moirai2".into()),
        );
        let mut seed = 1500u64;
        let s = 0.1;
        put_res(&mut w, "in_proj", 8, 4, 16, &mut seed, s);
        put2(&mut w, "blk.0.attn_q.weight", 16, 16, &mut seed, s);
        put2(&mut w, "blk.0.attn_k.weight", 16, 16, &mut seed, s);
        put2(&mut w, "blk.0.attn_v.weight", 16, 16, &mut seed, s);
        put1(&mut w, "blk.0.norm1.weight", 16, &mut seed);
        put1(&mut w, "blk.0.norm2.weight", 16, &mut seed);
        put2(&mut w, "blk.0.attn_o.weight", 16, 16, &mut seed, s);
        put1(&mut w, "blk.0.attn_qn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_kn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_vbias.weight", 4, &mut seed);
        put2(&mut w, "blk.0.ffn_fc1.weight", 32, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_fc2.weight", 16, 32, &mut seed, s);
        put2(&mut w, "blk.0.ffn_gate.weight", 32, 16, &mut seed, s);
        put1(&mut w, "norm_f.weight", 16, &mut seed);
        put_res(&mut w, "out_proj", 8, 16, 12, &mut seed, s);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::Moirai2Model::load(&path, cfg.clone()).unwrap();
        let burn = BurnMoirai2Model::load(&path, cfg).unwrap();
        // horizon 5 exercises prefill + one decode step.
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, 5).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx, 5).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let err: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        println!("moirai2 synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "moirai2 Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
