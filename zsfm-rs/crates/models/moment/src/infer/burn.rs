//! MOMENT-1-large inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): RevIN, patch embedding with mask
//! tokens, T5 relative-bias encoder blocks, gated-GELU FFN, reconstruction
//! head. Weight loading goes through the existing candle GGUF reader and
//! converts each F32 tensor to Burn `TensorData`, so there is exactly one
//! GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};

use crate::config::MomentConfig;

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

fn linear(x: Tensor<2>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<2> {
    match b {
        Some(bias) => zsfm_burn::linear::burn_linear_bias(x, w, bias),
        None => zsfm_burn::linear::burn_linear_nobias(x, w),
    }
}

fn rms(x: Tensor<2>, w: Tensor<1>, eps: f32) -> Tensor<2> {
    zsfm_burn::norm::burn_rms_norm(x, w, eps)
}

struct BurnEncoderBlock {
    attn_qkv_w: Tensor<2>,
    attn_o_w: Tensor<2>,
    attn_norm_w: Tensor<1>,
    ffn_wi0_w: Tensor<2>,
    ffn_wi1_w: Tensor<2>,
    ffn_wo_w: Tensor<2>,
    ffn_norm_w: Tensor<1>,
}

pub struct BurnMomentModel {
    config: MomentConfig,
    patch_embed_w: Tensor<2>,
    pos_embed: Tensor<3>,
    mask_embed: Tensor<1>,
    rel_bias_data: Vec<f32>,
    blocks: Vec<BurnEncoderBlock>,
    norm_f_w: Tensor<1>,
    head_w: Tensor<2>,
    head_b: Tensor<1>,
}

impl BurnMomentModel {
    pub fn load(gguf_path: &Path, config: MomentConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let patch_embed_w = load_t(&content, &mut reader, "patch_embed.weight")?;
        let pos_t = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "pos_embed.pe",
            &CDevice::Cpu,
            DType::F32,
        )?;
        let pos_dims = pos_t.dims().to_vec();
        anyhow::ensure!(pos_dims.len() == 3, "pos_embed must be 3D");
        let pos_data: Vec<f32> = pos_t.flatten_all()?.to_vec1()?;
        let pos_embed = Tensor::<3>::from_data(
            TensorData::new(pos_data, [pos_dims[0], pos_dims[1], pos_dims[2]]),
            &device(),
        );
        let mask_embed = load_v(&content, &mut reader, "mask_embed")?;
        let rel_bias_data: Vec<f32> = zsfm_nn::load_tensor(
            &content,
            &mut reader,
            "blk.0.attn_rel_bias.weight",
            &CDevice::Cpu,
            DType::F32,
        )?
        .flatten_all()?
        .to_vec1()?;

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
            let qd: Vec<f32> = qw.flatten_all()?.to_vec1()?;
            let kd: Vec<f32> = kw.flatten_all()?.to_vec1()?;
            let vd: Vec<f32> = vw.flatten_all()?.to_vec1()?;
            let d = config.d_model;
            let mut cat = Vec::with_capacity(3 * d * d);
            cat.extend_from_slice(&qd);
            cat.extend_from_slice(&kd);
            cat.extend_from_slice(&vd);
            let dev = device();
            let attn_qkv_w = Tensor::<2>::from_data(TensorData::new(cat, [3 * d, d]), &dev);
            blocks.push(BurnEncoderBlock {
                attn_qkv_w,
                attn_o_w: load_t(&content, &mut reader, &p("attn_o.weight"))?,
                attn_norm_w: load_v(&content, &mut reader, &p("attn_norm.weight"))?,
                ffn_wi0_w: load_t(&content, &mut reader, &p("ffn_wi0.weight"))?,
                ffn_wi1_w: load_t(&content, &mut reader, &p("ffn_wi1.weight"))?,
                ffn_wo_w: load_t(&content, &mut reader, &p("ffn_wo.weight"))?,
                ffn_norm_w: load_v(&content, &mut reader, &p("ffn_norm.weight"))?,
            });
        }

        let norm_f_w = load_v(&content, &mut reader, "norm_f.weight")?;
        let head_w = load_t(&content, &mut reader, "head.weight")?;
        let head_b = load_v(&content, &mut reader, "head.bias")?;

        Ok(Self {
            config,
            patch_embed_w,
            pos_embed,
            mask_embed,
            rel_bias_data,
            blocks,
            norm_f_w,
            head_w,
            head_b,
        })
    }

    pub fn forecast(&self, context: &[f32], horizon: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let (loc, scale) = revin_stats(context);
        let scale = scale.max(1e-8);
        let mut ctx_scaled: Vec<f32> = context.iter().map(|&v| (v - loc) / scale).collect();
        if ctx_scaled.len() < cfg.seq_len {
            let pad = cfg.seq_len - ctx_scaled.len();
            let mut padded = vec![0.0f32; pad];
            padded.extend_from_slice(&ctx_scaled);
            ctx_scaled = padded;
        } else if ctx_scaled.len() > cfg.seq_len {
            let start = ctx_scaled.len() - cfg.seq_len;
            ctx_scaled = ctx_scaled[start..].to_vec();
        }

        let ctx_patches = cfg.seq_len / cfg.patch_stride;
        let n_fc = (horizon + cfg.patch_len - 1) / cfg.patch_len;
        let total_patches = ctx_patches + n_fc;
        let rel_bias = self.compute_rel_bias(total_patches);

        let ctx_data = patchify(&ctx_scaled, cfg.patch_len, cfg.patch_stride, ctx_patches);
        let mut h = self.embed_patches_with_mask(&ctx_data, n_fc, total_patches);
        for blk in &self.blocks {
            h = self.forward_block(h, blk, rel_bias.clone(), total_patches);
        }
        h = rms(h, self.norm_f_w.clone(), cfg.layer_norm_eps as f32);
        let future_h: Tensor<2> = h.narrow(0, ctx_patches, n_fc);
        let pred = linear(future_h, self.head_w.clone(), Some(self.head_b.clone()));
        let pred_flat: Vec<f32> = pred
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(pred_flat
            .iter()
            .take(horizon)
            .map(|&v| v * scale + loc)
            .collect())
    }

    fn embed_patches_with_mask(
        &self,
        ctx_patches: &[Vec<f32>],
        n_fc: usize,
        total_patches: usize,
    ) -> Tensor<2> {
        let cfg = &self.config;
        let dev = device();
        let flat: Vec<f32> = ctx_patches.iter().flatten().copied().collect();
        let x = Tensor::<2>::from_data(
            TensorData::new(flat, [ctx_patches.len(), cfg.patch_len]),
            &dev,
        );
        let h_ctx = x.matmul(self.patch_embed_w.clone().transpose());
        let h_fc = self
            .mask_embed
            .clone()
            .unsqueeze_dim::<2>(0)
            .expand([n_fc, cfg.d_model]);
        let h = Tensor::cat(vec![h_ctx, h_fc], 0);
        let pos: Tensor<2> = self
            .pos_embed
            .clone()
            .squeeze_dim::<2>(0)
            .narrow(0, 0, total_patches);
        h + pos
    }

    fn forward_block(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        rel_bias: Tensor<4>,
        seq_len: usize,
    ) -> Tensor<2> {
        let h = rms(
            hidden.clone(),
            blk.attn_norm_w.clone(),
            self.config.layer_norm_eps as f32,
        );
        let h = self.t5_self_attn(h, blk, rel_bias, seq_len);
        let h = h + hidden;
        let h2 = rms(
            h.clone(),
            blk.ffn_norm_w.clone(),
            self.config.layer_norm_eps as f32,
        );
        let h2 = gated_gelu_ffn(h2, &blk.ffn_wi0_w, &blk.ffn_wi1_w, &blk.ffn_wo_w);
        h2 + h
    }

    fn t5_self_attn(
        &self,
        hidden: Tensor<2>,
        blk: &BurnEncoderBlock,
        rel_bias: Tensor<4>,
        seq_len: usize,
    ) -> Tensor<2> {
        let cfg = &self.config;
        let qkv = linear(hidden, blk.attn_qkv_w.clone(), None);
        let q: Tensor<2> = qkv.clone().narrow(1, 0, cfg.d_model);
        let k: Tensor<2> = qkv.clone().narrow(1, cfg.d_model, cfg.d_model);
        let v: Tensor<2> = qkv.narrow(1, 2 * cfg.d_model, cfg.d_model);
        let split = |t: Tensor<2>| {
            t.reshape([seq_len, cfg.n_heads, cfg.head_dim])
                .permute([1, 0, 2])
        };
        let (q, k, v) = (split(q), split(k), split(v));
        let scores = q.matmul(k.permute([0, 2, 1]));
        let rel: Tensor<3> = rel_bias.squeeze_dim(0);
        let scores = scores + rel;
        let attn = activation::softmax(scores, 2);
        let out = attn.matmul(v);
        let out: Tensor<2> = out.permute([1, 0, 2]).reshape([seq_len, cfg.d_model]);
        linear(out, blk.attn_o_w.clone(), None)
    }

    fn compute_rel_bias(&self, seq_len: usize) -> Tensor<4> {
        let cfg = &self.config;
        let mut out = vec![0.0f32; cfg.n_heads * seq_len * seq_len];
        for query_pos in 0..seq_len {
            for key_pos in 0..seq_len {
                let rel = key_pos as i64 - query_pos as i64;
                let bucket = t5_relative_bucket(
                    rel,
                    true,
                    cfg.rel_attn_num_buckets,
                    cfg.rel_attn_max_distance,
                );
                for head in 0..cfg.n_heads {
                    out[head * seq_len * seq_len + query_pos * seq_len + key_pos] =
                        self.rel_bias_data[bucket * cfg.n_heads + head];
                }
            }
        }
        Tensor::<4>::from_data(
            TensorData::new(out, [1, cfg.n_heads, seq_len, seq_len]),
            &device(),
        )
    }
}

fn gated_gelu_ffn(
    x: Tensor<2>,
    wi0_w: &Tensor<2>,
    wi1_w: &Tensor<2>,
    wo_w: &Tensor<2>,
) -> Tensor<2> {
    let gate = activation::gelu(linear(x.clone(), wi0_w.clone(), None));
    let value = linear(x, wi1_w.clone(), None);
    linear(gate * value, wo_w.clone(), None)
}

fn t5_relative_bucket(
    relative_position: i64,
    bidirectional: bool,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let mut ret = 0usize;
    let mut num_buckets = num_buckets;
    let n: usize = if bidirectional {
        num_buckets /= 2;
        if relative_position > 0 {
            ret += num_buckets;
        }
        relative_position.unsigned_abs() as usize
    } else {
        (-relative_position).max(0) as usize
    };
    let max_exact = num_buckets / 2;
    if n < max_exact {
        ret += n;
    } else {
        let val = max_exact
            + ((n as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln()
                * (num_buckets - max_exact) as f32) as usize;
        ret += val.min(num_buckets - 1);
    }
    ret
}

fn revin_stats(x: &[f32]) -> (f32, f32) {
    let n = x.len() as f64;
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean as f32, var.sqrt() as f32)
}

fn patchify(x: &[f32], patch_len: usize, stride: usize, num_patches: usize) -> Vec<Vec<f32>> {
    (0..num_patches)
        .map(|i| {
            let start = i * stride;
            let end = (start + patch_len).min(x.len());
            let mut patch = x[start..end].to_vec();
            patch.resize(patch_len, 0.0);
            patch
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

    fn tiny_config() -> MomentConfig {
        MomentConfig {
            d_model: 16,
            n_layers: 1,
            n_heads: 2,
            head_dim: 8,
            d_ff: 32,
            seq_len: 8,
            patch_len: 2,
            patch_stride: 2,
            num_patches: 4,
            rel_attn_num_buckets: 8,
            rel_attn_max_distance: 16,
            layer_norm_eps: 1e-6,
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

    fn put3(w: &mut GGUFWriter, name: &str, d0: usize, d1: usize, d2: usize, seed: &mut u64) {
        *seed += 1;
        let bytes: Vec<u8> = pseudo(d0 * d1 * d2, *seed)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.add_tensor(
            name,
            vec![d2 as u64, d1 as u64, d0 as u64],
            GGMLType::F32,
            bytes,
        );
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        let dir = std::env::temp_dir().join(format!("moment-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("moment".into()),
        );
        let mut seed = 500u64;
        put2(&mut w, "patch_embed.weight", 16, 2, &mut seed, 0.3);
        put3(&mut w, "pos_embed.pe", 1, 8, 16, &mut seed);
        put1(&mut w, "mask_embed", 16, &mut seed);
        put1(&mut w, "blk.0.attn_rel_bias.weight", 16, &mut seed);
        for n in ["attn_q", "attn_k", "attn_v", "attn_o"] {
            put2(&mut w, &format!("blk.0.{n}.weight"), 16, 16, &mut seed, 0.3);
        }
        put1(&mut w, "blk.0.attn_norm.weight", 16, &mut seed);
        put2(&mut w, "blk.0.ffn_wi0.weight", 32, 16, &mut seed, 0.3);
        put2(&mut w, "blk.0.ffn_wi1.weight", 32, 16, &mut seed, 0.3);
        put2(&mut w, "blk.0.ffn_wo.weight", 16, 32, &mut seed, 0.3);
        put1(&mut w, "blk.0.ffn_norm.weight", 16, &mut seed);
        put1(&mut w, "norm_f.weight", 16, &mut seed);
        put2(&mut w, "head.weight", 2, 16, &mut seed, 0.3);
        put1(&mut w, "head.bias", 2, &mut seed);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::MomentModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnMomentModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
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
        println!("moment synthetic: candle {candle_ms:.3}ms burn {burn_ms:.3}ms err {err:.2e}");
        assert!(err < 1e-4, "moment Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
