//! Chronos-2 inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle): patch embedding with time features,
//! encoder blocks (TimeSelfAttention with Llama RoPE + GroupSelfAttention +
//! FFN), quantile output head. Weight loading goes through the existing
//! candle GGUF reader and converts each F32 tensor to Burn `TensorData`,
//! so there is exactly one GGUF parser.

use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};
use zsfm_burn::linear::burn_linear_nd;
use zsfm_burn::norm::burn_rms_norm_nd;
use zsfm_burn::rope::{apply_llama_rope, rope_tables};

use super::InferConfig;

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

fn load_w(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
    expected_d_out: usize,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_weight(content, reader, name, expected_d_out, &CDevice::Cpu)?;
    burn2(&t)
}

fn linear<const D: usize>(x: Tensor<D>, w: Tensor<2>, b: Option<Tensor<1>>) -> Tensor<D> {
    burn_linear_nd(x, w, b)
}

fn rms<const D: usize>(x: Tensor<D>, w: Tensor<1>, eps: f64) -> Tensor<D> {
    burn_rms_norm_nd(x, w, eps as f32)
}

fn softmax_last<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    activation::softmax(x, D - 1)
}

fn apply_act<const D: usize>(x: Tensor<D>, name: &str) -> Result<Tensor<D>> {
    Ok(match name {
        "relu" => activation::relu(x),
        "gelu" => activation::gelu(x),
        "gelu_new" | "gelu_pytorch_tanh" => activation::gelu_approximate(x),
        "silu" | "swish" => activation::silu(x),
        other => bail!("unsupported activation function: {other}"),
    })
}

struct BurnResidualBlockW {
    hidden_w: Tensor<2>,
    hidden_b: Tensor<1>,
    output_w: Tensor<2>,
    output_b: Tensor<1>,
    skip_w: Tensor<2>,
    skip_b: Tensor<1>,
}

struct BurnAttnW {
    qkv_w: Tensor<2>,
    o_w: Tensor<2>,
    norm_w: Tensor<1>,
}

struct BurnFfnW {
    wi_w: Tensor<2>,
    wo_w: Tensor<2>,
    norm_w: Tensor<1>,
}

struct BurnBlockW {
    time_attn: BurnAttnW,
    group_attn: BurnAttnW,
    ffn: BurnFfnW,
}

pub struct BurnChronosModel {
    config: InferConfig,
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    token_embd: Tensor<2>,
    input_patch: BurnResidualBlockW,
    blocks: Vec<BurnBlockW>,
    enc_norm: Tensor<1>,
    output_patch: BurnResidualBlockW,
}

fn load_residual_block(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
    d_out_h: usize,
    d_out: usize,
) -> Result<BurnResidualBlockW> {
    Ok(BurnResidualBlockW {
        hidden_w: load_w(content, reader, &format!("{prefix}.hidden.weight"), d_out_h)?,
        hidden_b: load_v(content, reader, &format!("{prefix}.hidden.bias"))?,
        output_w: load_w(content, reader, &format!("{prefix}.output.weight"), d_out)?,
        output_b: load_v(content, reader, &format!("{prefix}.output.bias"))?,
        skip_w: load_w(content, reader, &format!("{prefix}.skip.weight"), d_out)?,
        skip_b: load_v(content, reader, &format!("{prefix}.skip.bias"))?,
    })
}

fn load_attn(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    blk: usize,
    kind: &str,
    d_model: usize,
    inner_dim: usize,
) -> Result<BurnAttnW> {
    let dev = device();
    let qw = zsfm_nn::load_weight(
        content,
        reader,
        &format!("blk.{blk}.{kind}.q.weight"),
        inner_dim,
        &CDevice::Cpu,
    )?;
    let kw = zsfm_nn::load_weight(
        content,
        reader,
        &format!("blk.{blk}.{kind}.k.weight"),
        inner_dim,
        &CDevice::Cpu,
    )?;
    let vw = zsfm_nn::load_weight(
        content,
        reader,
        &format!("blk.{blk}.{kind}.v.weight"),
        inner_dim,
        &CDevice::Cpu,
    )?;
    let mut d = Vec::new();
    for t in [&qw, &kw, &vw] {
        d.extend_from_slice(&t.flatten_all()?.to_vec1::<f32>()?);
    }
    let qkv_w = Tensor::<2>::from_data(TensorData::new(d, [3 * inner_dim, d_model]), &dev);
    Ok(BurnAttnW {
        qkv_w,
        o_w: load_w(
            content,
            reader,
            &format!("blk.{blk}.{kind}.o.weight"),
            d_model,
        )?,
        norm_w: load_v(content, reader, &format!("blk.{blk}.{kind}_norm.weight"))?,
    })
}

impl BurnChronosModel {
    pub fn patch_size(&self) -> usize {
        self.config.patch_size()
    }
    pub fn patch_stride(&self) -> usize {
        self.config.patch_stride()
    }
    pub fn context_length(&self) -> usize {
        self.config.context_length()
    }
    pub fn quantiles(&self) -> &[f32] {
        self.config.quantiles()
    }

    pub fn load(gguf_path: &Path, config: InferConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let d = config.d_model;
        let ff = config.d_ff;
        let id = config.num_heads * config.d_kv;
        let ps = config.patch_size;
        let nq = config.quantiles.len();

        let token_embd = load_t(&content, &mut reader, "token_embd.weight")?;
        let input_patch = load_residual_block(&content, &mut reader, "input_patch", ff, d)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for n in 0..config.num_layers {
            let time_attn = load_attn(&content, &mut reader, n, "time_attn", d, id)?;
            let group_attn = load_attn(&content, &mut reader, n, "group_attn", d, id)?;
            blocks.push(BurnBlockW {
                time_attn,
                group_attn,
                ffn: BurnFfnW {
                    wi_w: load_w(&content, &mut reader, &format!("blk.{n}.ffn.wi.weight"), ff)?,
                    wo_w: load_w(&content, &mut reader, &format!("blk.{n}.ffn.wo.weight"), d)?,
                    norm_w: load_v(&content, &mut reader, &format!("blk.{n}.ffn_norm.weight"))?,
                },
            });
        }

        let enc_norm = load_v(&content, &mut reader, "enc_norm.weight")?;
        let out_d = nq * ps;
        let output_patch = load_residual_block(&content, &mut reader, "output_patch", ff, out_d)?;

        let (rope_cos, rope_sin) = rope_tables(config.d_kv, 8192, config.rope_theta, &device())?;
        Ok(Self {
            config,
            rope_cos,
            rope_sin,
            token_embd,
            input_patch,
            blocks,
            enc_norm,
            output_patch,
        })
    }

    pub fn forecast(&self, context: &[f32], prediction_length: usize) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.config;
        let ps = cfg.patch_size;
        let stride = cfg.patch_stride;
        let n_quantiles = cfg.quantiles.len();
        let n_out_patches = (prediction_length + ps - 1) / ps;

        let (normalized, loc, scale) = instance_norm(context, cfg.use_arcsinh);
        let padded = pad_for_patching(&normalized, ps);
        let n_ctx_patches = (padded.len() - ps) / stride + 1;

        let ctx_features = build_patch_features(
            &padded,
            n_ctx_patches,
            ps,
            stride,
            -((n_ctx_patches * ps) as f32),
            0.0,
            cfg.time_encoding_scale as f32,
            true,
        );
        let fut_features =
            build_patch_features_future(n_out_patches, ps, 0.0, cfg.time_encoding_scale as f32);

        let dev = device();
        let ctx_tensor =
            Tensor::<2>::from_data(TensorData::new(ctx_features, [n_ctx_patches, 3 * ps]), &dev);
        let fut_tensor =
            Tensor::<2>::from_data(TensorData::new(fut_features, [n_out_patches, 3 * ps]), &dev);
        let ctx_embeds = self.forward_residual_block(ctx_tensor, &self.input_patch)?;
        let fut_embeds = self.forward_residual_block(fut_tensor, &self.input_patch)?;

        let seq = if cfg.use_reg_token {
            let reg: Tensor<2> = self.token_embd.clone().narrow(0, 1, 1);
            Tensor::cat(vec![ctx_embeds, reg, fut_embeds], 0)
        } else {
            Tensor::cat(vec![ctx_embeds, fut_embeds], 0)
        };
        let total_seq = seq.dims()[0];
        let hidden = seq.unsqueeze_dim::<3>(0);
        let hidden = self.forward_encoder(hidden, total_seq)?;
        let forecast_embeds: Tensor<3> = hidden.narrow(1, total_seq - n_out_patches, n_out_patches);
        let forecast_embeds: Tensor<2> = forecast_embeds.squeeze_dim(0);
        let quantile_raw = self.forward_residual_block(forecast_embeds, &self.output_patch)?;

        let total_out = n_out_patches * ps;
        let qmat: Tensor<3> = quantile_raw.reshape([n_out_patches, n_quantiles, ps]);
        let qmat: Tensor<3> = qmat.permute([1, 0, 2]);
        let qmat: Tensor<2> = qmat.reshape([n_quantiles, total_out]);
        let qmat: Tensor<2> = if total_out > prediction_length {
            qmat.narrow(1, 0, prediction_length)
        } else {
            qmat
        };
        let data: Vec<f32> = qmat
            .to_data()
            .try_to_vec()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut result = vec![vec![0.0f32; prediction_length]; n_quantiles];
        for q in 0..n_quantiles {
            for t in 0..prediction_length {
                let v = data[q * prediction_length + t] as f64;
                let v = if cfg.use_arcsinh { v.sinh() } else { v };
                result[q][t] = (v as f32) * scale + loc;
            }
        }
        Ok(result)
    }

    fn forward_encoder(&self, mut x: Tensor<3>, seq_len: usize) -> Result<Tensor<3>> {
        for blk in &self.blocks {
            x = self.forward_block(x, blk, seq_len)?;
        }
        Ok(rms(x, self.enc_norm.clone(), self.config.layer_norm_eps))
    }

    fn forward_block(&self, x: Tensor<3>, blk: &BurnBlockW, seq_len: usize) -> Result<Tensor<3>> {
        let normed = rms(
            x.clone(),
            blk.time_attn.norm_w.clone(),
            self.config.layer_norm_eps,
        );
        let attn_out = self.forward_time_attn(normed, &blk.time_attn, seq_len);
        let x = x + attn_out;
        let normed = rms(
            x.clone(),
            blk.group_attn.norm_w.clone(),
            self.config.layer_norm_eps,
        );
        let grp_out = self.forward_group_attn_univariate(normed, &blk.group_attn);
        let x = x + grp_out;
        let normed = rms(
            x.clone(),
            blk.ffn.norm_w.clone(),
            self.config.layer_norm_eps,
        );
        let ffn_out = self.forward_ffn(normed, &blk.ffn)?;
        Ok(x + ffn_out)
    }

    fn forward_time_attn(&self, x: Tensor<3>, w: &BurnAttnW, seq_len: usize) -> Tensor<3> {
        let id = self.config.num_heads * self.config.d_kv;
        let nh = self.config.num_heads;
        let dkv = self.config.d_kv;
        let qkv = linear(x, w.qkv_w.clone(), None);
        let q: Tensor<3> = qkv.clone().narrow(2, 0, id);
        let k: Tensor<3> = qkv.clone().narrow(2, id, id);
        let v: Tensor<3> = qkv.narrow(2, 2 * id, id);
        let pack = |t: Tensor<3>| t.reshape([1, seq_len, nh, dkv]).permute([0, 2, 1, 3]);
        let (q, k, v) = (pack(q), pack(k), pack(v));
        let q = apply_llama_rope(q, self.rope_cos.clone(), self.rope_sin.clone(), 0);
        let k = apply_llama_rope(k, self.rope_cos.clone(), self.rope_sin.clone(), 0);
        let scores = q.matmul(k.transpose());
        let attn = softmax_last(scores);
        let out = attn.matmul(v);
        let out: Tensor<3> = out.permute([0, 2, 1, 3]).reshape([1, seq_len, id]);
        linear(out, w.o_w.clone(), None)
    }

    fn forward_group_attn_univariate(&self, x: Tensor<3>, w: &BurnAttnW) -> Tensor<3> {
        let id = self.config.num_heads * self.config.d_kv;
        let qkv = linear(x, w.qkv_w.clone(), None);
        let v: Tensor<3> = qkv.narrow(2, 2 * id, id);
        linear(v, w.o_w.clone(), None)
    }

    fn forward_ffn(&self, x: Tensor<3>, w: &BurnFfnW) -> Result<Tensor<3>> {
        let h = linear(x, w.wi_w.clone(), None);
        let h = apply_act(h, &self.config.dense_act_fn)?;
        Ok(linear(h, w.wo_w.clone(), None))
    }

    fn forward_residual_block(&self, x: Tensor<2>, w: &BurnResidualBlockW) -> Result<Tensor<2>> {
        let h = linear(x.clone(), w.hidden_w.clone(), Some(w.hidden_b.clone()));
        let h = apply_act(h, &self.config.dense_act_fn)?;
        let out = linear(h, w.output_w.clone(), Some(w.output_b.clone()));
        let skip = linear(x, w.skip_w.clone(), Some(w.skip_b.clone()));
        Ok(out + skip)
    }
}

fn instance_norm(x: &[f32], use_arcsinh: bool) -> (Vec<f32>, f32, f32) {
    let n = x.len() as f64;
    let loc = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - loc).powi(2)).sum::<f64>() / n;
    let scale = (var.sqrt() as f32).max(1e-5);
    let mut out: Vec<f32> = x.iter().map(|&v| (v as f64 - loc) as f32 / scale).collect();
    if use_arcsinh {
        for v in &mut out {
            *v = (*v as f64).asinh() as f32;
        }
    }
    (out, loc as f32, scale)
}

fn pad_for_patching(x: &[f32], patch_size: usize) -> Vec<f32> {
    let rem = x.len() % patch_size;
    if rem == 0 {
        x.to_vec()
    } else {
        let pad_len = patch_size - rem;
        let mut out = vec![f32::NAN; pad_len];
        out.extend_from_slice(x);
        out
    }
}

fn build_patch_features(
    padded: &[f32],
    n_patches: usize,
    patch_size: usize,
    stride: usize,
    time_start: f32,
    _time_end: f32,
    time_scale: f32,
    observed: bool,
) -> Vec<f32> {
    let feature_dim = 3 * patch_size;
    let mut out = vec![0.0f32; n_patches * feature_dim];
    for p in 0..n_patches {
        let offset = p * stride;
        let base = p * feature_dim;
        for i in 0..patch_size {
            let t = offset + i;
            let global_t = time_start + t as f32;
            out[base + i] = global_t / time_scale;
            let v = padded.get(t).copied().unwrap_or(0.0);
            let is_obs = observed && v.is_finite();
            out[base + patch_size + i] = if is_obs { v } else { 0.0 };
            out[base + 2 * patch_size + i] = if is_obs { 1.0 } else { 0.0 };
        }
    }
    out
}

fn build_patch_features_future(
    n_patches: usize,
    patch_size: usize,
    time_start: f32,
    time_scale: f32,
) -> Vec<f32> {
    let feature_dim = 3 * patch_size;
    let mut out = vec![0.0f32; n_patches * feature_dim];
    for p in 0..n_patches {
        let offset = p * patch_size;
        let base = p * feature_dim;
        for i in 0..patch_size {
            let global_t = time_start + (offset + i) as f32;
            out[base + i] = global_t / time_scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Chronos2Config, ChronosInnerConfig};
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

    fn tiny_config() -> InferConfig {
        let c2 = Chronos2Config {
            d_model: 16,
            d_kv: 8,
            d_ff: 32,
            num_layers: 1,
            num_heads: 2,
            layer_norm_epsilon: 1e-6,
            rope_theta: 10000.0,
            feed_forward_proj: "relu".into(),
            chronos_config: ChronosInnerConfig {
                context_length: 64,
                input_patch_size: 4,
                output_patch_size: 4,
                input_patch_stride: 2,
                quantiles: vec![0.1, 0.5, 0.9],
                use_reg_token: false,
                use_arcsinh: false,
                max_output_patches: 1,
                time_encoding_scale: None,
            },
        };
        InferConfig::from(&c2)
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
        d_out: usize,
        seed: &mut u64,
        s: f32,
    ) {
        put2(w, &format!("{prefix}.hidden.weight"), h, d_in, seed, s);
        put1(w, &format!("{prefix}.hidden.bias"), h, seed);
        put2(w, &format!("{prefix}.output.weight"), d_out, h, seed, s);
        put1(w, &format!("{prefix}.output.bias"), d_out, seed);
        put2(w, &format!("{prefix}.skip.weight"), d_out, d_in, seed, s);
        put1(w, &format!("{prefix}.skip.bias"), d_out, seed);
    }

    fn put_attn(
        w: &mut GGUFWriter,
        blk: usize,
        kind: &str,
        d: usize,
        id: usize,
        seed: &mut u64,
        s: f32,
    ) {
        for t in ["q", "k", "v"] {
            put2(w, &format!("blk.{blk}.{kind}.{t}.weight"), id, d, seed, s);
        }
        put2(w, &format!("blk.{blk}.{kind}.o.weight"), d, id, seed, s);
        put1(w, &format!("blk.{blk}.{kind}_norm.weight"), d, seed);
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        // d=16, id=16, ff=32, ps=4, stride=2, 3 quantiles, 1 layer.
        let dir = std::env::temp_dir().join(format!("chronos-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        let mut w = GGUFWriter::new();
        w.add_metadata(
            "general.architecture",
            GGUFMetaValue::String("chronos".into()),
        );
        let mut seed = 900u64;
        let s = 0.2;
        put2(&mut w, "token_embd.weight", 4, 16, &mut seed, s);
        put_res(&mut w, "input_patch", 32, 12, 16, &mut seed, s);
        put_attn(&mut w, 0, "time_attn", 16, 16, &mut seed, s);
        put_attn(&mut w, 0, "group_attn", 16, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn.wi.weight", 32, 16, &mut seed, s);
        put2(&mut w, "blk.0.ffn.wo.weight", 16, 32, &mut seed, s);
        put1(&mut w, "blk.0.ffn_norm.weight", 16, &mut seed);
        put1(&mut w, "enc_norm.weight", 16, &mut seed);
        put_res(&mut w, "output_patch", 32, 16, 12, &mut seed, s);
        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = tiny_config();
        let candle = crate::ChronosModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnChronosModel::load(&path, cfg).unwrap();
        let ctx: Vec<f32> = (0..8).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx, 4).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx, 4).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(a.len(), b.len());
        let mut err = 0.0f32;
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                err = err.max((x - y).abs());
            }
        }
        println!("chronos synthetic: candle {candle_ms:.1}ms burn {burn_ms:.1}ms err {err:.2e}");
        assert!(err < 1e-4, "chronos Burn parity failed: {err:.2e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
