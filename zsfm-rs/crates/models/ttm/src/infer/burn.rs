//! TinyTimeMixer inference on the Burn backend (Flex CPU).
//!
//! Bit-exact port of `super` (candle) `forecast`: same StdScaler, patchify,
//! mixer math, and inverse scale. Weight loading goes through the existing
//! candle GGUF reader and converts each F32 tensor to Burn `TensorData`,
//! so there is exactly one GGUF parser. The parity test below builds a
//! synthetic GGUF, loads it through both engines, and asserts agreement.

use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::{activation, Device, Tensor, TensorData};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device as CDevice};

use crate::config::TtmConfig;

type BDev = Device;

fn device() -> BDev {
    Device::flex()
}

/// Burn tensor from a candle F32 tensor with identical values and shape.
fn from_candle(t: &candle_core::Tensor) -> Result<Tensor<2>> {
    let dims = t.dims().to_vec();
    anyhow::ensure!(dims.len() == 2, "expected 2D weight, got {dims:?}");
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(Tensor::<2>::from_data(
        TensorData::new(data, [dims[0], dims[1]]),
        &device(),
    ))
}

fn from_candle_1(t: &candle_core::Tensor) -> Result<Tensor<1>> {
    let data: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(Tensor::<1>::from_data(
        TensorData::new(data, [t.elem_count()]),
        &device(),
    ))
}

fn load_t(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<2>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    from_candle(&t)
}

fn load_v(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    name: &str,
) -> Result<Tensor<1>> {
    let t = zsfm_nn::load_tensor(content, reader, name, &CDevice::Cpu, DType::F32)?;
    from_candle_1(&t)
}

struct BurnMixerLayer {
    patch_norm_w: Tensor<1>,
    patch_norm_b: Tensor<1>,
    patch_fc1_w: Tensor<2>,
    patch_fc1_b: Tensor<1>,
    patch_fc2_w: Tensor<2>,
    patch_fc2_b: Tensor<1>,
    patch_gate_w: Tensor<2>,
    patch_gate_b: Tensor<1>,
    feat_norm_w: Tensor<1>,
    feat_norm_b: Tensor<1>,
    feat_fc1_w: Tensor<2>,
    feat_fc1_b: Tensor<1>,
    feat_fc2_w: Tensor<2>,
    feat_fc2_b: Tensor<1>,
    feat_gate_w: Tensor<2>,
    feat_gate_b: Tensor<1>,
}

struct BurnAdaptiveLevel {
    factor: usize,
    layers: Vec<BurnMixerLayer>,
}

pub struct BurnTtmModel {
    config: TtmConfig,
    patcher_w: Tensor<2>,
    patcher_b: Tensor<1>,
    enc_levels: Vec<BurnAdaptiveLevel>,
    dec_adapter_w: Tensor<2>,
    dec_adapter_b: Tensor<1>,
    dec_layers: Vec<BurnMixerLayer>,
    head_w: Tensor<2>,
    head_b: Tensor<1>,
}

fn load_mixer_layer(
    content: &gguf_file::Content,
    reader: &mut (impl std::io::Read + std::io::Seek),
    prefix: &str,
) -> Result<BurnMixerLayer> {
    Ok(BurnMixerLayer {
        patch_norm_w: load_v(content, reader, &format!("{prefix}.patch_norm.weight"))?,
        patch_norm_b: load_v(content, reader, &format!("{prefix}.patch_norm.bias"))?,
        patch_fc1_w: load_t(content, reader, &format!("{prefix}.patch_fc1.weight"))?,
        patch_fc1_b: load_v(content, reader, &format!("{prefix}.patch_fc1.bias"))?,
        patch_fc2_w: load_t(content, reader, &format!("{prefix}.patch_fc2.weight"))?,
        patch_fc2_b: load_v(content, reader, &format!("{prefix}.patch_fc2.bias"))?,
        patch_gate_w: load_t(content, reader, &format!("{prefix}.patch_gate.weight"))?,
        patch_gate_b: load_v(content, reader, &format!("{prefix}.patch_gate.bias"))?,
        feat_norm_w: load_v(content, reader, &format!("{prefix}.feat_norm.weight"))?,
        feat_norm_b: load_v(content, reader, &format!("{prefix}.feat_norm.bias"))?,
        feat_fc1_w: load_t(content, reader, &format!("{prefix}.feat_fc1.weight"))?,
        feat_fc1_b: load_v(content, reader, &format!("{prefix}.feat_fc1.bias"))?,
        feat_fc2_w: load_t(content, reader, &format!("{prefix}.feat_fc2.weight"))?,
        feat_fc2_b: load_v(content, reader, &format!("{prefix}.feat_fc2.bias"))?,
        feat_gate_w: load_t(content, reader, &format!("{prefix}.feat_gate.weight"))?,
        feat_gate_b: load_v(content, reader, &format!("{prefix}.feat_gate.bias"))?,
    })
}

fn linear(x: Tensor<2>, w: Tensor<2>, b: Tensor<1>) -> Tensor<2> {
    zsfm_burn::linear::burn_linear_bias(x, w, b)
}

fn layer_norm(x: Tensor<2>, w: Tensor<1>, b: Tensor<1>, eps: f32) -> Tensor<2> {
    zsfm_burn::norm::burn_layer_norm(x, w, b, eps)
}

fn gelu(x: Tensor<2>) -> Tensor<2> {
    activation::gelu(x)
}

fn softmax_last_dim(x: Tensor<2>) -> Tensor<2> {
    activation::softmax(x, 1)
}

impl BurnTtmModel {
    pub fn load(gguf_path: &Path, config: TtmConfig) -> Result<Self> {
        let file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let mut reader = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
        let content = gguf_file::Content::read(&mut reader).context("parse GGUF header")?;

        let patcher_w = load_t(&content, &mut reader, "enc.patcher.weight")?;
        let patcher_b = load_v(&content, &mut reader, "enc.patcher.bias")?;

        let n_levels = config.adaptive_patching_levels;
        let mut enc_levels = Vec::with_capacity(n_levels);
        for l in 0..n_levels {
            let factor = 1usize << (n_levels - 1 - l);
            let mut layers = Vec::with_capacity(config.num_layers);
            for n in 0..config.num_layers {
                layers.push(load_mixer_layer(
                    &content,
                    &mut reader,
                    &format!("enc.blk.{l}.layer.{n}"),
                )?);
            }
            enc_levels.push(BurnAdaptiveLevel { factor, layers });
        }

        let dec_adapter_w = load_t(&content, &mut reader, "dec.adapter.weight")?;
        let dec_adapter_b = load_v(&content, &mut reader, "dec.adapter.bias")?;
        let mut dec_layers = Vec::with_capacity(config.decoder_num_layers);
        for n in 0..config.decoder_num_layers {
            dec_layers.push(load_mixer_layer(
                &content,
                &mut reader,
                &format!("dec.blk.{n}"),
            )?);
        }
        let head_w = load_t(&content, &mut reader, "head.weight")?;
        let head_b = load_v(&content, &mut reader, "head.bias")?;

        Ok(Self {
            config,
            patcher_w,
            patcher_b,
            enc_levels,
            dec_adapter_w,
            dec_adapter_b,
            dec_layers,
            head_w,
            head_b,
        })
    }

    pub fn forecast(&self, context: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let (scaled, mean, std) = std_scale(context);
        let patches = patchify(&scaled, cfg.patch_length, cfg.patch_stride, cfg.num_patches);
        let flat: Vec<f32> = patches.into_iter().flatten().collect();
        let dev = device();
        let mut h = Tensor::<2>::from_data(
            TensorData::new(flat, [cfg.num_patches, cfg.patch_length]),
            &dev,
        );

        h = linear(h, self.patcher_w.clone(), self.patcher_b.clone());
        for level in &self.enc_levels {
            h = self.forward_adaptive_level(h, level);
        }
        h = linear(h, self.dec_adapter_w.clone(), self.dec_adapter_b.clone());
        for layer in &self.dec_layers {
            h = forward_mixer_layer(h, layer, cfg.norm_eps as f32);
        }
        let flat_dim = cfg.num_patches * cfg.decoder_d_model;
        let h = h.reshape([flat_dim]);
        let h = h.unsqueeze_dim::<2>(0);
        let h = linear(h, self.head_w.clone(), self.head_b.clone());
        let h: Tensor<1> = h.squeeze_dim(0);
        let forecast: Vec<f32> = h.to_data().try_to_vec()?;
        Ok(forecast.iter().map(|&v| v * std + mean).collect())
    }

    fn forward_adaptive_level(&self, hidden: Tensor<2>, level: &BurnAdaptiveLevel) -> Tensor<2> {
        let factor = level.factor;
        let [p, f] = hidden.dims();
        let mut h = if factor > 1 {
            hidden.reshape([p * factor, f / factor])
        } else {
            hidden
        };
        for layer in &level.layers {
            h = forward_mixer_layer(h, layer, self.config.norm_eps as f32);
        }
        if factor > 1 {
            h.reshape([p, f])
        } else {
            h
        }
    }
}

fn forward_mixer_layer(h: Tensor<2>, layer: &BurnMixerLayer, eps: f32) -> Tensor<2> {
    forward_feat_mixer(forward_patch_mixer(h.clone(), layer, eps), layer, eps)
}

fn forward_patch_mixer(h: Tensor<2>, w: &BurnMixerLayer, eps: f32) -> Tensor<2> {
    let normed = layer_norm(
        h.clone(),
        w.patch_norm_w.clone(),
        w.patch_norm_b.clone(),
        eps,
    );
    let t = normed.transpose();
    let m = gelu(linear(t, w.patch_fc1_w.clone(), w.patch_fc1_b.clone()));
    let m = linear(m, w.patch_fc2_w.clone(), w.patch_fc2_b.clone());
    let gate = softmax_last_dim(linear(
        m.clone(),
        w.patch_gate_w.clone(),
        w.patch_gate_b.clone(),
    ));
    let gated = m * gate;
    gated.transpose() + h
}

fn forward_feat_mixer(h: Tensor<2>, w: &BurnMixerLayer, eps: f32) -> Tensor<2> {
    let normed = layer_norm(h.clone(), w.feat_norm_w.clone(), w.feat_norm_b.clone(), eps);
    let m = gelu(linear(normed, w.feat_fc1_w.clone(), w.feat_fc1_b.clone()));
    let m = linear(m, w.feat_fc2_w.clone(), w.feat_fc2_b.clone());
    let gate = softmax_last_dim(linear(
        m.clone(),
        w.feat_gate_w.clone(),
        w.feat_gate_b.clone(),
    ));
    let gated = m * gate;
    gated + h
}

fn std_scale(x: &[f32]) -> (Vec<f32>, f32, f32) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt() as f32;
    let scaled: Vec<f32> = x.iter().map(|&v| (v as f32 - mean as f32) / std).collect();
    (scaled, mean as f32, std)
}

fn patchify(
    x: &[f32],
    patch_length: usize,
    patch_stride: usize,
    num_patches: usize,
) -> Vec<Vec<f32>> {
    let new_seq_len = patch_length + patch_stride * (num_patches - 1);
    let padded: Vec<f32> = if x.len() < new_seq_len {
        let pad_val = x.first().copied().unwrap_or(0.0);
        let mut v = vec![pad_val; new_seq_len - x.len()];
        v.extend_from_slice(x);
        v
    } else {
        x[x.len() - new_seq_len..].to_vec()
    };
    (0..num_patches)
        .map(|i| padded[i * patch_stride..i * patch_stride + patch_length].to_vec())
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

    fn test_config() -> TtmConfig {
        TtmConfig {
            context_length: 64,
            prediction_length: 4,
            patch_length: 4,
            patch_stride: 2,
            d_model: 8,
            num_layers: 1,
            decoder_d_model: 8,
            decoder_num_layers: 1,
            expansion_factor: 2,
            adaptive_patching_levels: 1,
            num_patches: 4,
            scaling: "std".into(),
            gated_attn: true,
            norm_eps: 1e-5,
        }
    }

    /// Write logical [R,C] row-major data with GGUF (reversed) shape [C,R].
    fn put2(w: &mut GGUFWriter, name: &str, rows: usize, cols: usize, seed: &mut u64) {
        *seed += 1;
        let data = pseudo(rows * cols, *seed);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(name, vec![cols as u64, rows as u64], GGMLType::F32, bytes);
    }

    fn put1(w: &mut GGUFWriter, name: &str, n: usize, seed: &mut u64) {
        *seed += 1;
        let data = pseudo(n, *seed);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor(name, vec![n as u64], GGMLType::F32, bytes);
    }

    fn put_layer(w: &mut GGUFWriter, prefix: &str, p: usize, f: usize, seed: &mut u64) {
        put1(w, &format!("{prefix}.patch_norm.weight"), f, seed);
        put1(w, &format!("{prefix}.patch_norm.bias"), f, seed);
        put2(w, &format!("{prefix}.patch_fc1.weight"), 2 * p, p, seed);
        put1(w, &format!("{prefix}.patch_fc1.bias"), 2 * p, seed);
        put2(w, &format!("{prefix}.patch_fc2.weight"), p, 2 * p, seed);
        put1(w, &format!("{prefix}.patch_fc2.bias"), p, seed);
        put2(w, &format!("{prefix}.patch_gate.weight"), p, p, seed);
        put1(w, &format!("{prefix}.patch_gate.bias"), p, seed);
        put1(w, &format!("{prefix}.feat_norm.weight"), f, seed);
        put1(w, &format!("{prefix}.feat_norm.bias"), f, seed);
        put2(w, &format!("{prefix}.feat_fc1.weight"), 2 * f, f, seed);
        put1(w, &format!("{prefix}.feat_fc1.bias"), 2 * f, seed);
        put2(w, &format!("{prefix}.feat_fc2.weight"), f, 2 * f, seed);
        put1(w, &format!("{prefix}.feat_fc2.bias"), f, seed);
        put2(w, &format!("{prefix}.feat_gate.weight"), f, f, seed);
        put1(w, &format!("{prefix}.feat_gate.bias"), f, seed);
    }

    #[test]
    fn burn_matches_candle_on_synthetic_gguf() {
        let dir = std::env::temp_dir().join(format!("ttm-burn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");

        let mut w = GGUFWriter::new();
        w.add_metadata("general.architecture", GGUFMetaValue::String("ttm".into()));
        let mut seed = 100u64;
        // P=4 patches, F=8 features.
        put2(&mut w, "enc.patcher.weight", 8, 4, &mut seed);
        put1(&mut w, "enc.patcher.bias", 8, &mut seed);
        put_layer(&mut w, "enc.blk.0.layer.0", 4, 8, &mut seed);
        put2(&mut w, "dec.adapter.weight", 8, 8, &mut seed);
        put1(&mut w, "dec.adapter.bias", 8, &mut seed);
        put_layer(&mut w, "dec.blk.0", 4, 8, &mut seed);
        put2(&mut w, "head.weight", 4, 32, &mut seed);
        put1(&mut w, "head.bias", 4, &mut seed);

        let f = std::fs::File::create(&path).unwrap();
        w.write_to(&mut std::io::BufWriter::new(f)).unwrap();

        let cfg = test_config();
        let candle = crate::TtmModel::load(&path, cfg.clone()).unwrap();
        let burn = BurnTtmModel::load(&path, cfg).unwrap();

        let ctx: Vec<f32> = (0..16).map(|i| 10.0 + 0.5 * i as f32).collect();
        let t0 = std::time::Instant::now();
        let a = candle.forecast(&ctx).unwrap();
        let candle_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = std::time::Instant::now();
        let b = burn.forecast(&ctx).unwrap();
        let burn_ms = t0.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(a.len(), b.len());
        let err: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        println!("ttm synthetic: candle {candle_ms:.3}ms burn {burn_ms:.3}ms err {err:.2e}");
        assert!(err < 1e-4, "ttm Burn parity failed: {err:.2e}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
