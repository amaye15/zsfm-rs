use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::Context;
use candle_core::{pickle, DType, Device};
use indicatif::{ProgressBar, ProgressStyle};

use zsfm_gguf::{GGMLType, GGUFMetaValue, GGUFWriter};

use crate::config::LagLlamaConfig;
use crate::tensor_map::map_tensor_name;

pub struct ConvertOptions {
    pub output_dtype: GGMLType,
}

/// Convert a Lag-Llama PyTorch Lightning checkpoint (.ckpt) to GGUF.
pub fn convert(
    ckpt_path: &Path,
    config: &LagLlamaConfig,
    opts: &ConvertOptions,
    output_path: &Path,
) -> anyhow::Result<()> {
    let mut writer = GGUFWriter::new();
    write_metadata(&mut writer, config);

    println!("Reading checkpoint {} …", ckpt_path.display());
    // PyTorch Lightning checkpoints store model weights under the "state_dict" key.
    let tensors = pickle::read_all_with_key(ckpt_path, Some("state_dict"))
        .with_context(|| format!("read checkpoint {}", ckpt_path.display()))?;

    let total = tensors.len();
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let device = Device::Cpu;
    let mut mapped = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut fallback_count = 0usize;

    for (hf_name, tensor) in &tensors {
        pb.set_message(hf_name.clone());

        let gguf_name = match map_tensor_name(hf_name) {
            Some(n) => n,
            None => {
                skipped.push(hf_name.clone());
                pb.inc(1);
                continue;
            }
        };

        // Shape from candle is in Python order (e.g. [out, in]).
        let py_shape: Vec<usize> = tensor.shape().dims().to_vec();
        let n_elems: usize = py_shape.iter().product();
        let innermost = py_shape.last().copied().unwrap_or(1);

        // Decode to f32 using candle — handles BF16/F16/F32 transparently.
        let f32_vals: Vec<f32> = tensor
            .to_device(&device)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;

        let (dst_dtype, gguf_shape, tensor_data) =
            if opts.output_dtype == GGMLType::Q8_0 && (innermost % 32 != 0 || n_elems % 32 != 0) {
                fallback_count += 1;
                let gs = py_shape.iter().rev().map(|&d| d as u64).collect();
                (GGMLType::F32, gs, f32_to_bytes(&f32_vals))
            } else {
                let (dst, data) = apply_dtype(&f32_vals, opts.output_dtype)?;
                let gs = py_shape.iter().rev().map(|&d| d as u64).collect();
                (dst, gs, data)
            };

        writer.add_tensor(gguf_name, gguf_shape, dst_dtype, tensor_data);
        mapped += 1;
        pb.inc(1);
    }

    pb.finish_with_message("tensors processed");

    if !skipped.is_empty() {
        eprintln!("\nWarning: {} tensor(s) skipped (unrecognised names):", skipped.len());
        for name in &skipped { eprintln!("  {name}"); }
    }
    if fallback_count > 0 {
        eprintln!("\nNote: {fallback_count} tensor(s) fell back to F32.");
    }

    println!("Writing {mapped} tensors to {} …", output_path.display());
    let out_file = File::create(output_path)
        .with_context(|| format!("create {}", output_path.display()))?;
    let mut buf_writer = BufWriter::new(out_file);
    writer.write_to(&mut buf_writer)?;
    println!("Done.");
    Ok(())
}

fn write_metadata(writer: &mut GGUFWriter, config: &LagLlamaConfig) {
    writer.add_metadata("general.architecture",   GGUFMetaValue::String("lag_llama".into()));
    writer.add_metadata("general.name",           GGUFMetaValue::String("Lag-Llama".into()));
    writer.add_metadata("lag_llama.n_layer",      GGUFMetaValue::Uint32(config.n_layer as u32));
    writer.add_metadata("lag_llama.n_head",       GGUFMetaValue::Uint32(config.n_head as u32));
    writer.add_metadata("lag_llama.n_embd_per_head", GGUFMetaValue::Uint32(config.n_embd_per_head as u32));
    writer.add_metadata("lag_llama.n_embd",       GGUFMetaValue::Uint32(config.n_embd as u32));
    writer.add_metadata("lag_llama.mlp_hidden",   GGUFMetaValue::Uint32(config.mlp_hidden as u32));
    writer.add_metadata("lag_llama.feature_size", GGUFMetaValue::Uint32(config.feature_size as u32));
    writer.add_metadata("lag_llama.n_lags",       GGUFMetaValue::Uint32(config.n_lags as u32));
    writer.add_metadata("lag_llama.n_time_feat",  GGUFMetaValue::Uint32(config.n_time_feat as u32));
    writer.add_metadata("lag_llama.max_context_length", GGUFMetaValue::Uint32(config.max_context_length as u32));
}

fn f32_to_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn apply_dtype(f32_vals: &[f32], dst: GGMLType) -> anyhow::Result<(GGMLType, Vec<u8>)> {
    match dst {
        GGMLType::F32 => Ok((GGMLType::F32, f32_to_bytes(f32_vals))),
        GGMLType::F16 => {
            let bytes: Vec<u8> = f32_vals.iter()
                .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
                .collect();
            Ok((GGMLType::F16, bytes))
        }
        GGMLType::BF16 => {
            let bytes: Vec<u8> = f32_vals.iter()
                .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            Ok((GGMLType::BF16, bytes))
        }
        GGMLType::Q8_0 => Ok((GGMLType::Q8_0, quantize_q8_0(f32_vals)?)),
    }
}

fn quantize_q8_0(values: &[f32]) -> anyhow::Result<Vec<u8>> {
    const BLOCK: usize = 32;
    if values.len() % BLOCK != 0 {
        anyhow::bail!("Q8_0 requires count divisible by {BLOCK}, got {}", values.len());
    }
    let n_blocks = values.len() / BLOCK;
    let mut out = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let blk = &values[b * BLOCK..(b + 1) * BLOCK];
        let amax = blk.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let d = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
        let d_inv = if d == 0.0 { 0.0f32 } else { 1.0 / d };
        let base = b * 34;
        out[base..base + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for i in 0..BLOCK {
            out[base + 2 + i] = (blk[i] * d_inv).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
    Ok(out)
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exp == 0xFF { return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }; }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return sign | 0x7C00; }
    if new_exp <= 0 {
        if new_exp < -10 { return sign; }
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp);
        return sign | (m >> 13) as u16;
    }
    sign | ((new_exp as u16) << 10) | (mantissa >> 13) as u16
}
