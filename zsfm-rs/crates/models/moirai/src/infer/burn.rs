//! Moirai-1.0 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): mean-scale norm, patch embedding,
//! masked bidirectional encoder with QK-norm attention, split-half RoPE,
//! variate bias, SwiGLU FFN, Student-t loc head. Weight loading goes through
//! the existing candle GGUF reader and converts each F32 tensor to Burn
//! `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::{burn_linear_bias, burn_linear_nobias};
use zsfm_burn::norm::burn_rms_norm_nd;

use crate::config::MoiraiConfig;

const PATCH_SIZE: usize = 32;
const PATCH_IDX: usize = 2;

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

fn load_3d(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<3>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    zsfm_burn::burn_weight_3d(&t, &device())
}

fn linear(x: Tensor<2>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<2> {
    match b {
        Some(bias) => burn_linear_bias(x, w, bias),
        None => burn_linear_nobias(x, w),
    }
}

fn linear_nd<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    zsfm_burn::linear::burn_linear_nd(x, w, b)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>) -> Tensor<D> {
    burn_rms_norm_nd(x, w, 1e-6)
}

fn swiglu_ffn(x: Tensor<2>, fc1_w: Tensor<2>, fc2_w: Tensor<2>, gate_w: Tensor<2>) -> Tensor<2> {
    let content = activation::silu(linear(x.clone(), fc1_w, None));
    let gate = linear(x, gate_w, None);
    linear(content * gate, fc2_w, None)
}

struct BurnEncoderBlock {
    norm1_w: Tensor<1>,
    norm2_w: Tensor<1>,
    attn_qkv_w: Tensor<2>,
    attn_o_w: Tensor<2>,
    attn_qn_w: Tensor<1>,
    attn_kn_w: Tensor<1>,
    vbias_obs: Tensor<3>,
    vbias_mask: Tensor<3>,
    ffn_fc1_w: Tensor<2>,
    ffn_fc2_w: Tensor<2>,
    ffn_gate_w: Tensor<2>,
}

pub struct BurnMoiraiModel {
    config: MoiraiConfig,
    in_proj_w: Tensor<3>,
    in_proj_b: Tensor<2>,
    mask_embed: Tensor<2>,
    blocks: Vec<BurnEncoderBlock>,
    norm_f_w: Tensor<1>,
    head_st_loc_w: Tensor<3>,
    head_st_loc_b: Tensor<2>,
    rope_inv_freq: Vec<f32>,
}

impl BurnMoiraiModel {
    pub fn load(gguf_path: &Path, config: MoiraiConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;
        let dev = device();

        let in_proj_w = load_3d(&content, &mut reader, "in_proj.weight")?;
        let in_proj_b = {
            let t = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                "in_proj.bias",
                &CDevice::Cpu,
                DType::F32,
            )?;
            let d: Vec<f32> = t.flatten_all()?.to_vec1()?;
            anyhow::ensure!(d.len() % 5 == 0, "in_proj.bias len");
            let d_model = d.len() / 5;
            Tensor::<2>::from_data(TensorData::new(d, [5, d_model]), &dev)
        };
        let mask_embed = load_t(&content, &mut reader, "mask_embed.weight")?;

        let n_heads = config.n_heads;
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
            let mut cat = Vec::new();
            for t in [&qw, &kw, &vw] {
                cat.extend_from_slice(&t.flatten_all()?.to_vec1::<f32>()?);
            }
            let d_model = config.d_model;
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
            let vbias_obs = Tensor::<3>::from_data(
                TensorData::new(vbias_raw[0..n_heads].to_vec(), [n_heads, 1, 1]),
                &dev,
            );
            let vbias_mask = Tensor::<3>::from_data(
                TensorData::new(vbias_raw[n_heads..2 * n_heads].to_vec(), [n_heads, 1, 1]),
                &dev,
            );
            blocks.push(BurnEncoderBlock {
                norm1_w: load_v(&content, &mut reader, &p("norm1.weight"))?,
                norm2_w: load_v(&content, &mut reader, &p("norm2.weight"))?,
                attn_qkv_w,
                attn_o_w: load_t(&content, &mut reader, &p("attn_o.weight"))?,
                attn_qn_w: load_v(&content, &mut reader, &p("attn_qn.weight"))?,
                attn_kn_w: load_v(&content, &mut reader, &p("attn_kn.weight"))?,
                vbias_obs,
                vbias_mask,
                ffn_fc1_w: load_t(&content, &mut reader, &p("ffn_fc1.weight"))?,
                ffn_fc2_w: load_t(&content, &mut reader, &p("ffn_fc2.weight"))?,
                ffn_gate_w: load_t(&content, &mut reader, &p("ffn_gate.weight"))?,
            });
        }

        let norm_f_w = load_v(&content, &mut reader, "norm_f.weight")?;
        let head_st_loc_w = load_3d(&content, &mut reader, "head.st_loc.weight")?;
        let head_st_loc_b = {
            let t = zsfm_nn::load_tensor(
                &content,
                &mut reader,
                "head.st_loc.bias",
                &CDevice::Cpu,
                DType::F32,
            )?;
            let d: Vec<f32> = t.flatten_all()?.to_vec1()?;
            anyhow::ensure!(d.len() % 5 == 0, "head bias len");
            let max_ps = d.len() / 5;
            Tensor::<2>::from_data(TensorData::new(d, [5, max_ps]), &dev)
        };

        let half = config.head_dim / 2;
        let rope_inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / 10000_f32.powf(2.0 * i as f32 / config.head_dim as f32))
            .collect();

        Ok(Self {
            config,
            in_proj_w,
            in_proj_b,
            mask_embed,
            blocks,
            norm_f_w,
            head_st_loc_w,
            head_st_loc_b,
            rope_inv_freq,
        })
    }

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let patch_size = PATCH_SIZE;
        let patch_idx = PATCH_IDX;
        let loc = context.iter().map(|&v| v as f64).sum::<f64>() / context.len() as f64;
        let scale =
            context.iter().map(|&v| (v as f64 - loc).abs()).sum::<f64>() / context.len() as f64;
        let scale = (scale.max(1e-8)) as f32;
        let loc = loc as f32;
        let max_ts = cfg.max_seq_len;
        let mut ctx_scaled: Vec<f32> = context.iter().map(|&v| (v - loc) / scale).collect();
        if ctx_scaled.len() > max_ts {
            let start = ctx_scaled.len() - max_ts;
            ctx_scaled = ctx_scaled[start..].to_vec();
        }
        let ctx_len = ctx_scaled.len();
        let ctx_padded_len = ((ctx_len + patch_size - 1) / patch_size) * patch_size;
        if ctx_padded_len > ctx_len {
            let mut padded = vec![0.0f32; ctx_padded_len - ctx_len];
            padded.extend_from_slice(&ctx_scaled);
            ctx_scaled = padded;
        }
        let n_ctx_patches = ctx_scaled.len() / patch_size;
        let n_fc_patches = (horizon + patch_size - 1) / patch_size;
        let total_patches = n_ctx_patches + n_fc_patches;
        let d_model = cfg.d_model;
        let max_ps = cfg.max_patch_size;
        let dev = device();

        let proj_w_full: Tensor<2> = self
            .in_proj_w
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<2>(0);
        let proj_w: Tensor<2> = proj_w_full.narrow(1, 0, patch_size);
        let proj_b: Tensor<1> = self
            .in_proj_b
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<1>(0);
        let mut patch_flat = vec![0.0f32; n_ctx_patches * patch_size];
        for i in 0..n_ctx_patches {
            let src = &ctx_scaled[i * patch_size..(i + 1) * patch_size];
            patch_flat[i * patch_size..(i + 1) * patch_size].copy_from_slice(src);
        }
        let ctx_patches_t = Tensor::<2>::from_data(
            TensorData::new(patch_flat, [n_ctx_patches, patch_size]),
            &dev,
        );
        let ctx_emb = linear(ctx_patches_t, proj_w, Some(proj_b));
        let mask_embed: Tensor<2> = self.mask_embed.clone().reshape([1, d_model]);
        let fc_emb = mask_embed.expand([n_fc_patches, d_model]);
        let mut h = Tensor::cat(vec![ctx_emb, fc_emb], 0);

        let mut is_masked = vec![0u8; total_patches];
        for i in n_ctx_patches..total_patches {
            is_masked[i] = 1;
        }
        for blk in &self.blocks {
            h = self.forward_block(h, blk, &is_masked, total_patches)?;
        }
        h = rms(h, self.norm_f_w.clone());

        let loc_w_3d: Tensor<2> = self
            .head_st_loc_w
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<2>(0);
        let loc_w: Tensor<2> = loc_w_3d.narrow(1, 0, patch_size);
        let loc_b: Tensor<1> = self
            .head_st_loc_b
            .clone()
            .narrow(0, patch_idx, 1)
            .squeeze_dim::<1>(0)
            .narrow(0, 0, patch_size);
        let future_h: Tensor<2> = h.narrow(0, n_ctx_patches, n_fc_patches);
        let pred = linear(future_h, loc_w, Some(loc_b));
        let pred_flat: Vec<f32> = pred
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let _ = max_ps;
        Ok(pred_flat
            .iter()
            .take(horizon)
            .map(|&v| v * scale + loc)
            .collect())
    }

    fn forward_block(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor<2>> {
        let h = rms(hidden.clone(), blk.norm1_w.clone());
        let h = self.qk_attn(h, blk, is_masked, seq_len)?;
        let h = h + hidden;
        let h2 = rms(h.clone(), blk.norm2_w.clone());
        let h2 = swiglu_ffn(
            h2,
            blk.ffn_fc1_w.clone(),
            blk.ffn_fc2_w.clone(),
            blk.ffn_gate_w.clone(),
        );
        Ok(h2 + h)
    }

    fn qk_attn(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        is_masked: &[u8],
        seq_len: usize,
    ) -> Result<Tensor<2>> {
        let cfg = &self.config;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let d_model = cfg.d_model;
        let qkv = linear_nd(hidden, blk.attn_qkv_w.clone(), None);
        let q: Tensor<2> = qkv.clone().narrow(1, 0, d_model);
        let k: Tensor<2> = qkv.clone().narrow(1, d_model, d_model);
        let v: Tensor<2> = qkv.narrow(1, 2 * d_model, d_model);
        let q: Tensor<3> = q.reshape([seq_len, n_heads, head_dim]);
        let k: Tensor<3> = k.reshape([seq_len, n_heads, head_dim]);
        let q = qk_norm_heads(q, blk.attn_qn_w.clone(), seq_len, n_heads, head_dim);
        let k = qk_norm_heads(k, blk.attn_kn_w.clone(), seq_len, n_heads, head_dim);
        let q: Tensor<3> = q.permute([1, 0, 2]);
        let k: Tensor<3> = k.permute([1, 0, 2]);
        let (cos_t, sin_t) = self.rope_tables(seq_len);
        let q = apply_rope_with_tables(q, cos_t.clone(), sin_t.clone(), head_dim);
        let k = apply_rope_with_tables(k, cos_t, sin_t, head_dim);
        let v: Tensor<3> = v.reshape([seq_len, n_heads, head_dim]).permute([1, 0, 2]);
        let scores = q
            .matmul(k.permute([0, 2, 1]))
            .mul_scalar(1.0 / (head_dim as f32).sqrt());
        let scores = apply_var_attn_bias(
            scores,
            is_masked,
            seq_len,
            blk.vbias_obs.clone(),
            blk.vbias_mask.clone(),
        );
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v);
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([seq_len, d_model]);
        Ok(linear_nd(out, blk.attn_o_w.clone(), None))
    }

    fn rope_tables(&self, seq_len: usize) -> (Tensor<3>, Tensor<3>) {
        let dev = device();
        let half = self.rope_inv_freq.len();
        let mut cos_v = vec![0.0f32; seq_len * half];
        let mut sin_v = vec![0.0f32; seq_len * half];
        for pos in 0..seq_len {
            for i in 0..half {
                let theta = pos as f32 * self.rope_inv_freq[i];
                cos_v[pos * half + i] = theta.cos();
                sin_v[pos * half + i] = theta.sin();
            }
        }
        let cos_t = Tensor::<2>::from_data(TensorData::new(cos_v, [seq_len, half]), &dev)
            .unsqueeze_dim::<3>(0);
        let sin_t = Tensor::<2>::from_data(TensorData::new(sin_v, [seq_len, half]), &dev)
            .unsqueeze_dim::<3>(0);
        (cos_t, sin_t)
    }
}

fn qk_norm_heads(
    x: Tensor<3>,
    weight: Tensor<1>,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor<3> {
    let x_flat: Tensor<2> = x.reshape([seq_len * n_heads, head_dim]);
    burn_rms_norm_nd(x_flat, weight, 1e-6).reshape([seq_len, n_heads, head_dim])
}

fn apply_var_attn_bias(
    scores: Tensor<3>,
    is_masked: &[u8],
    seq_len: usize,
    vbias_obs: Tensor<3>,
    vbias_mask: Tensor<3>,
) -> Tensor<3> {
    let dev = device();
    let mask_f: Vec<f32> = is_masked.iter().map(|&m| m as f32).collect();
    let mask_t = Tensor::<3>::from_data(TensorData::new(mask_f, [1, 1, seq_len]), &dev);
    let delta = vbias_mask - vbias_obs.clone();
    let bias = vbias_obs + delta * mask_t;
    scores + bias
}

fn apply_rope_with_tables(
    x: Tensor<3>,
    cos_t: Tensor<3>,
    sin_t: Tensor<3>,
    head_dim: usize,
) -> Tensor<3> {
    let half = head_dim / 2;
    let x1 = x.clone().narrow(2, 0, half);
    let x2 = x.clone().narrow(2, half, half);
    let rot1 = x1.clone() * cos_t.clone() - x2.clone() * sin_t.clone();
    let rot2 = x1 * sin_t + x2 * cos_t;
    Tensor::cat(vec![rot1, rot2], 2)
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

    fn tiny_config() -> MoiraiConfig {
        MoiraiConfig {
            d_model: 32,
            n_layers: 1,
            n_heads: 4,
            head_dim: 8,
            d_ff: 64,
            max_seq_len: 64,
            patch_sizes: vec![8, 16, 32, 64, 128],
            max_patch_size: 32,
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

    fn put3(
        w: &mut GGUFWriter,
        name: &str,
        d0: usize,
        d1: usize,
        d2: usize,
        seed: &mut u64,
        scale: f32,
    ) {
        *seed += 1;
        let data: Vec<f32> = pseudo(d0 * d1 * d2, *seed)
            .iter()
            .map(|v| v * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(
            name,
            vec![d2 as u64, d1 as u64, d0 as u64],
            GGMLType::F32,
            bytes,
        );
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        // d=32, heads=4, hd=8, 1 layer, patch 32.
        let dir = std::env::temp_dir().join(format!("moirai-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("moirai".into()),
        );
        let mut seed = 1300u64;
        let s = 0.1;
        put3(&mut w, "in_proj.weight", 5, 32, 32, &mut seed, s);
        put2(&mut w, "in_proj.bias", 5, 32, &mut seed, 1.0);
        put2(&mut w, "mask_embed.weight", 1, 32, &mut seed, 1.0);
        put2(&mut w, "blk.0.attn_q.weight", 32, 32, &mut seed, s);
        put2(&mut w, "blk.0.attn_k.weight", 32, 32, &mut seed, s);
        put2(&mut w, "blk.0.attn_v.weight", 32, 32, &mut seed, s);
        put1(&mut w, "blk.0.norm1.weight", 32, &mut seed);
        put1(&mut w, "blk.0.norm2.weight", 32, &mut seed);
        put2(&mut w, "blk.0.attn_o.weight", 32, 32, &mut seed, s);
        put1(&mut w, "blk.0.attn_qn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_kn.weight", 8, &mut seed);
        put1(&mut w, "blk.0.attn_vbias.weight", 8, &mut seed);
        put2(&mut w, "blk.0.ffn_fc1.weight", 64, 32, &mut seed, s);
        put2(&mut w, "blk.0.ffn_fc2.weight", 32, 64, &mut seed, s);
        put2(&mut w, "blk.0.ffn_gate.weight", 64, 32, &mut seed, s);
        put1(&mut w, "norm_f.weight", 32, &mut seed);
        put3(&mut w, "head.st_loc.weight", 5, 32, 32, &mut seed, s);
        put2(&mut w, "head.st_loc.bias", 5, 32, &mut seed, 1.0);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::MoiraiModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnMoiraiModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..32).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, 4).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx, 4).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let err: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        println!("moirai synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "moirai Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
