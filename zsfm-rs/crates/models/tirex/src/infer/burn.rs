//! TiRex inference on the Burn backend (Flex CPU) for the two tensor
//! stages, with all host kernels reused verbatim from `super`.
//!
//! TiRex is already ~90% engine-agnostic host code (sLSTM recurrence,
//! patching, scaling, SIMD kernels). Only `residual_block_forward` and
//! `ffn_forward` touch tensors, so this port re-implements just those two
//! on Burn and reuses every host helper from the parent module via
//! `super::` (children see parent-private items). Weight loading goes
//! through the existing candle GGUF reader and converts each F32 tensor to
//! Burn `TensorData`, so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;

use super::{
    adjust_context, compute_ry, headwise_linear_batch_ng, rms_norm_inplace, simd_gate_update,
    standard_scaler,
};
use crate::config::TiRexConfig;

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

fn load_vec(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Vec<f32>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    Ok(t.flatten_all()?.to_vec1()?)
}

fn to_host<const D: usize>(t: &Tensor<D>) -> Result<Vec<f32>> {
    t.to_data()
        .try_to_vec()
        .map_err(|e| anyhow::anyhow!("{e:?}"))
}

struct BurnEmbedBlock {
    hidden_w: Tensor<2>,
    hidden_b: Vec<f32>,
    output_w: Tensor<2>,
    output_b: Vec<f32>,
    residual_w: Tensor<2>,
    residual_b: Vec<f32>,
}

struct BurnBlock {
    norm_slstm: Vec<f32>,
    fizo_w: Vec<f32>,
    slstm_kernel_t: Vec<f32>,
    slstm_bias: Vec<f32>,
    group_norm_w: Vec<f32>,
    norm_ffn: Vec<f32>,
    ffn_gate_w: Tensor<2>,
    ffn_up_w: Tensor<2>,
    ffn_down_w: Tensor<2>,
}

pub struct BurnTiRexModel {
    config: TiRexConfig,
    in_emb: BurnEmbedBlock,
    blocks: Vec<BurnBlock>,
    out_norm: Vec<f32>,
    out_emb: BurnEmbedBlock,
}

fn load_embed_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnEmbedBlock> {
    let p = |s: &str| format!("{prefix}.{s}");
    Ok(BurnEmbedBlock {
        hidden_w: load_t(content, reader, &p("hidden.weight"))?,
        hidden_b: load_vec(content, reader, &p("hidden.bias"))?,
        output_w: load_t(content, reader, &p("output.weight"))?,
        output_b: load_vec(content, reader, &p("output.bias"))?,
        residual_w: load_t(content, reader, &p("residual.weight"))?,
        residual_b: load_vec(content, reader, &p("residual.bias"))?,
    })
}

impl BurnTiRexModel {
    pub fn load(gguf_path: &Path, config: TiRexConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let in_emb = load_embed_block(&content, &mut reader, "in_emb")?;
        let nh = config.num_heads;
        let dh = config.head_dim();
        let wpp = dh * dh;
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for n in 0..config.num_blocks {
            let p = |s: &str| format!("blk.{n}.{s}");
            let fgate_w = load_vec(&content, &mut reader, &p("fgate.weight"))?;
            let igate_w = load_vec(&content, &mut reader, &p("igate.weight"))?;
            let zgate_w = load_vec(&content, &mut reader, &p("zgate.weight"))?;
            let ogate_w = load_vec(&content, &mut reader, &p("ogate.weight"))?;
            let mut fizo_w = vec![0.0f32; nh * 4 * wpp];
            for h in 0..nh {
                fizo_w[h * 4 * wpp..h * 4 * wpp + wpp]
                    .copy_from_slice(&fgate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + wpp..h * 4 * wpp + 2 * wpp]
                    .copy_from_slice(&igate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + 2 * wpp..h * 4 * wpp + 3 * wpp]
                    .copy_from_slice(&zgate_w[h * wpp..(h + 1) * wpp]);
                fizo_w[h * 4 * wpp + 3 * wpp..h * 4 * wpp + 4 * wpp]
                    .copy_from_slice(&ogate_w[h * wpp..(h + 1) * wpp]);
            }
            let raw_kernel = load_vec(&content, &mut reader, &p("slstm_kernel"))?;
            let ng = 4usize;
            let mut slstm_kernel_t = vec![0.0f32; nh * ng * dh * dh];
            for head in 0..nh {
                for gate_d in 0..(ng * dh) {
                    for di in 0..dh {
                        slstm_kernel_t[head * ng * dh * dh + gate_d * dh + di] =
                            raw_kernel[head * dh * ng * dh + di * ng * dh + gate_d];
                    }
                }
            }
            blocks.push(BurnBlock {
                norm_slstm: load_vec(&content, &mut reader, &p("norm_slstm"))?,
                fizo_w,
                slstm_kernel_t,
                slstm_bias: load_vec(&content, &mut reader, &p("slstm_bias"))?,
                group_norm_w: load_vec(&content, &mut reader, &p("group_norm"))?,
                norm_ffn: load_vec(&content, &mut reader, &p("norm_ffn"))?,
                ffn_gate_w: load_t(&content, &mut reader, &p("ffn_gate.weight"))?,
                ffn_up_w: load_t(&content, &mut reader, &p("ffn_up.weight"))?,
                ffn_down_w: load_t(&content, &mut reader, &p("ffn_down.weight"))?,
            });
        }
        let out_norm = load_vec(&content, &mut reader, "out_norm")?;
        let out_emb = load_embed_block(&content, &mut reader, "out_emb")?;
        Ok(Self {
            config,
            in_emb,
            blocks,
            out_norm,
            out_emb,
        })
    }
}

/// Burn residual MLP block over host row-major data.
fn residual_block_forward_burn(
    x: &[f32],
    s: usize,
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
    emb: &BurnEmbedBlock,
) -> Vec<f32> {
    let dev = device();
    let xt = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [s, in_dim]), &dev);
    let bh = Tensor::<1>::from_data(TensorData::new(emb.hidden_b.clone(), [h_dim]), &dev);
    let h = activation::relu(burn_linear_nd(xt.clone(), emb.hidden_w.clone(), Some(bh)));
    let bo = Tensor::<1>::from_data(TensorData::new(emb.output_b.clone(), [out_dim]), &dev);
    let out = burn_linear_nd(h, emb.output_w.clone(), Some(bo));
    let br = Tensor::<1>::from_data(TensorData::new(emb.residual_b.clone(), [out_dim]), &dev);
    let res = burn_linear_nd(xt, emb.residual_w.clone(), Some(br));
    let y: Vec<f32> = (out + res).to_data().try_to_vec().unwrap();
    y
}

/// Burn SiLU-gated FFN over host row-major data.
fn ffn_forward_burn(
    x: &[f32],
    s: usize,
    in_dim: usize,
    up_dim: usize,
    blk: &BurnBlock,
) -> Vec<f32> {
    let dev = device();
    let xt = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [s, in_dim]), &dev);
    let gate = burn_linear_nd(xt.clone(), blk.ffn_gate_w.clone(), None);
    let up = burn_linear_nd(xt, blk.ffn_up_w.clone(), None);
    let h = activation::silu(gate) * up;
    let out = burn_linear_nd(h, blk.ffn_down_w.clone(), None);
    out.to_data().try_to_vec().unwrap()
}

impl BurnTiRexModel {
    pub fn forecast(
        &self,
        context: &[f32],
        prediction_length: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let cfg = &self.config;
        let patch_size = cfg.patch_size;
        let n_quantiles = cfg.num_quantiles;
        let median_idx = cfg
            .quantiles
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(4);
        let n_steps = prediction_length.div_ceil(patch_size);
        let mut all_q: Vec<f32> = Vec::with_capacity(n_steps * patch_size * n_quantiles);
        let mut ctx: Vec<f32> = context.to_vec();
        for _ in 0..n_steps {
            let full_ctx = adjust_context(&ctx, cfg.train_ctx_len);
            let (loc, scale) = standard_scaler(&full_ctx);
            let s_vals: Vec<f32> = full_ctx
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        0.0f32
                    } else {
                        (x - loc) / scale
                    }
                })
                .collect();
            let s_mask: Vec<f32> = full_ctx
                .iter()
                .map(|&x| if x.is_nan() { 0.0f32 } else { 1.0f32 })
                .collect();
            let num_patches = cfg.train_ctx_len / patch_size;
            let mut patched_vals = vec![0.0f32; num_patches * patch_size];
            let mut patched_mask = vec![0.0f32; num_patches * patch_size];
            for p in 0..num_patches {
                let start = p * patch_size;
                patched_vals[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_vals[start..start + patch_size]);
                patched_mask[p * patch_size..(p + 1) * patch_size]
                    .copy_from_slice(&s_mask[start..start + patch_size]);
            }
            let in_dim = patch_size * 2;
            let mut x_in = vec![0.0f32; num_patches * in_dim];
            for p in 0..num_patches {
                x_in[p * in_dim..p * in_dim + patch_size]
                    .copy_from_slice(&patched_vals[p * patch_size..(p + 1) * patch_size]);
                x_in[p * in_dim + patch_size..p * in_dim + in_dim]
                    .copy_from_slice(&patched_mask[p * patch_size..(p + 1) * patch_size]);
            }
            let mut hidden = residual_block_forward_burn(
                &x_in,
                num_patches,
                in_dim,
                cfg.input_ff_dim,
                cfg.embedding_dim,
                &self.in_emb,
            );
            let ng = 4usize;
            let d = cfg.embedding_dim;
            let mut sc_xg = vec![0.0f32; num_patches * ng * d];
            let mut sc_hout = vec![0.0f32; num_patches * d];
            let mut sc_y = vec![0.0f32; num_patches * d];
            let mut sc_xn = vec![0.0f32; num_patches * d];
            let mut sc_raw = vec![0.0f32; ng * d];
            let mut sc_ry_raw = vec![0.0f32; ng * d];
            let mut sc_ry_out = vec![0.0f32; ng * d];
            let mut sc_hnew = vec![0.0f32; d];
            let mut sc_cnew = vec![0.0f32; d];
            let mut sc_nnew = vec![0.0f32; d];
            let mut sc_mnew = vec![0.0f32; d];
            for block in &self.blocks {
                hidden = self.forward_slstm_block(
                    &hidden,
                    num_patches,
                    block,
                    &mut sc_xg,
                    &mut sc_hout,
                    &mut sc_y,
                    &mut sc_xn,
                    &mut sc_raw,
                    &mut sc_ry_raw,
                    &mut sc_ry_out,
                    &mut sc_hnew,
                    &mut sc_cnew,
                    &mut sc_nnew,
                    &mut sc_mnew,
                )?;
            }
            rms_norm_inplace(&mut hidden, &self.out_norm, cfg.embedding_dim, 1e-6);
            let out_dim = cfg.output_dim();
            let preds = residual_block_forward_burn(
                &hidden,
                num_patches,
                cfg.embedding_dim,
                cfg.input_ff_dim,
                out_dim,
                &self.out_emb,
            );
            let last = &preds[(num_patches - 1) * out_dim..num_patches * out_dim];
            for q in 0..n_quantiles {
                for d in 0..patch_size {
                    let v = last[q * patch_size + d] * scale + loc;
                    all_q.push(v);
                }
            }
            ctx.extend(std::iter::repeat(f32::NAN).take(patch_size));
        }
        let total = n_steps * patch_size;
        let mut quantiles: Vec<Vec<f32>> = vec![Vec::with_capacity(prediction_length); n_quantiles];
        for t in 0..prediction_length.min(total) {
            let step = t / patch_size;
            let d = t % patch_size;
            for q in 0..n_quantiles {
                quantiles[q].push(all_q[step * n_quantiles * patch_size + q * patch_size + d]);
            }
        }
        let median: Vec<f32> = (0..prediction_length)
            .map(|t| quantiles[median_idx][t])
            .collect();
        Ok((quantiles, median))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_slstm_block(
        &self,
        x: &[f32],
        s: usize,
        blk: &BurnBlock,
        sc_xg: &mut [f32],
        sc_hout: &mut [f32],
        sc_y: &mut [f32],
        sc_xn: &mut [f32],
        sc_raw: &mut [f32],
        sc_ry_raw: &mut [f32],
        sc_ry_out: &mut [f32],
        sc_hnew: &mut Vec<f32>,
        sc_cnew: &mut Vec<f32>,
        sc_nnew: &mut Vec<f32>,
        sc_mnew: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let d = cfg.embedding_dim;
        let nh = cfg.num_heads;
        let dh = cfg.head_dim();
        let ng = 4usize;
        sc_xn[..s * d].copy_from_slice(&x[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_slstm, d, 1e-6);
        headwise_linear_batch_ng(&sc_xn[..s * d], &blk.fizo_w, s, nh, dh, ng, sc_xg);
        let mut h = vec![0.0f32; d];
        let mut c = vec![0.0f32; d];
        let mut n = vec![0.0f32; d];
        let mut m = vec![f32::NEG_INFINITY; d];
        for t in 0..s {
            let wx = &sc_xg[t * ng * d..(t + 1) * ng * d];
            compute_ry(&h, &blk.slstm_kernel_t, nh, dh, ng, sc_ry_raw, sc_ry_out);
            for i in 0..ng * d {
                sc_raw[i] = wx[i] + sc_ry_out[i] + blk.slstm_bias[i];
            }
            let is_first = t == 0;
            simd_gate_update(
                &sc_raw[..d],
                &sc_raw[d..2 * d],
                &sc_raw[2 * d..3 * d],
                &sc_raw[3 * d..],
                &c,
                &n,
                &m,
                sc_cnew,
                sc_nnew,
                sc_hnew,
                sc_mnew,
                is_first,
            );
            std::mem::swap(&mut h, sc_hnew);
            std::mem::swap(&mut c, sc_cnew);
            std::mem::swap(&mut n, sc_nnew);
            std::mem::swap(&mut m, sc_mnew);
            sc_hout[t * d..(t + 1) * d].copy_from_slice(&h);
        }
        for t in 0..s {
            let h_t = &sc_hout[t * d..(t + 1) * d];
            for head in 0..nh {
                let start = head * dh;
                let h_slice = &h_t[start..start + dh];
                let mean = h_slice.iter().sum::<f32>() / dh as f32;
                let var = h_slice
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f32>()
                    / dh as f32;
                let inv_std = 1.0 / (var + 1e-5f32).sqrt();
                let w_slice = &blk.group_norm_w[start..start + dh];
                for d_i in 0..dh {
                    sc_y[t * d + start + d_i] =
                        (h_t[start + d_i] - mean) * inv_std * (1.0 + w_slice[d_i]);
                }
            }
        }
        let mut x_out = x.to_vec();
        for i in 0..s * d {
            x_out[i] += sc_y[i];
        }
        sc_xn[..s * d].copy_from_slice(&x_out[..s * d]);
        rms_norm_inplace(&mut sc_xn[..s * d], &blk.norm_ffn, d, 1e-6);
        let ffn_out = ffn_forward_burn(&sc_xn[..s * d], s, d, cfg.ffn_up_dim, blk);
        for i in 0..s * d {
            x_out[i] += ffn_out[i];
        }
        Ok(x_out)
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

    fn tiny_config() -> TiRexConfig {
        TiRexConfig {
            patch_size: 4,
            num_blocks: 1,
            embedding_dim: 16,
            num_heads: 2,
            input_ff_dim: 32,
            ffn_up_dim: 24,
            train_ctx_len: 16,
            quantiles: vec![0.25, 0.5, 0.75],
            num_quantiles: 3,
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

    fn put_emb(
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
        let dir = std::env::temp_dir().join(format!("tirex-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("tirex".into()),
        );
        let mut seed = 2900u64;
        let s = 0.05;
        put_emb(&mut w, "in_emb", 32, 8, 16, &mut seed, s);
        for t in ["fgate", "igate", "zgate", "ogate"] {
            put2(&mut w, &format!("blk.0.{t}.weight"), 16, 16, &mut seed, s);
        }
        // slstm_kernel logical [NH, DH, NG*DH] = [2,8,32].
        put1(&mut w, "blk.0.slstm_kernel", 2 * 8 * 32, &mut seed);
        // NOTE: raw kernel values unscaled (structure, not numerics, matters here).
        put1(&mut w, "blk.0.slstm_bias", 64, &mut seed);
        put1(&mut w, "blk.0.norm_slstm", 16, &mut seed);
        put1(&mut w, "blk.0.group_norm", 16, &mut seed);
        put1(&mut w, "blk.0.norm_ffn", 16, &mut seed);
        put2(&mut w, "blk.0.ffn_gate.weight", 24, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_up.weight", 24, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn_down.weight", 16, 24, &mut seed, s);
        put1(&mut w, "out_norm", 16, &mut seed);
        put_emb(&mut w, "out_emb", 32, 16, 12, &mut seed, s);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::TiRexModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnTiRexModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let (aq, am) = candle.forecast(&ctx, 2).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let (bq, bm) = burn.forecast(&ctx, 2).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut err = 0.0f32;
        for (ra, rb) in aq.iter().zip(bq.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                err = err.max((x - y).abs());
            }
        }
        for (x, y) in am.iter().zip(bm.iter()) {
            err = err.max((x - y).abs());
        }
        println!("tirex synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-3, "tirex Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
