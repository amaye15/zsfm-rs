use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::Context;
use indicatif::{ProgressBar, ProgressStyle};
use safetensors::SafeTensors;
use safetensors::Dtype as StDtype;

use crate::config::TtmConfig;
use crate::download::ModelFiles;
use crate::gguf::{GGMLType, GGUFMetaValue, GGUFWriter};
use crate::tensor_map::map_tensor_name;

pub struct ConvertOptions {
    pub output_dtype: GGMLType,
}

pub fn convert(
    model_id: &str,
    files: &ModelFiles,
    config: &TtmConfig,
    opts: &ConvertOptions,
    output_path: &Path,
) -> anyhow::Result<()> {
    let mut writer = GGUFWriter::new();
    write_metadata(&mut writer, model_id, config);

    let shard_bytes = load_shard_bytes(&files.safetensors_shards)?;
    let shard_views: Vec<SafeTensors> = shard_bytes
        .iter()
        .map(|b| SafeTensors::deserialize(b).context("deserialize shard"))
        .collect::<anyhow::Result<_>>()?;

    let total_tensors: usize = shard_views.iter().map(|s| s.len()).sum();
    println!("Found {} tensors across {} shard(s).", total_tensors, shard_views.len());

    let pb = ProgressBar::new(total_tensors as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let mut mapped = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut fallback_count = 0usize;

    for shard in &shard_views {
        for (hf_name, tensor_view) in shard.tensors() {
            pb.set_message(hf_name.to_string());

            let gguf_name = match map_tensor_name(&hf_name) {
                Some(n) => n,
                None => {
                    skipped.push(hf_name.to_string());
                    pb.inc(1);
                    continue;
                }
            };

            let src_dtype = ggml_type_from_st(tensor_view.dtype())
                .with_context(|| format!("tensor {hf_name}: unsupported dtype {:?}", tensor_view.dtype()))?;

            let raw_data = tensor_view.data();
            let py_shape = tensor_view.shape();
            let n_elems: usize = py_shape.iter().product();
            let innermost = py_shape.last().copied().unwrap_or(1);
            let outermost = py_shape.first().copied().unwrap_or(1);

            let (dst_dtype, gguf_shape, tensor_data) =
                if opts.output_dtype == GGMLType::Q8_0 && (innermost % 32 != 0 || n_elems % 32 != 0) {
                    fallback_count += 1;
                    let data = cast_data(raw_data, src_dtype, GGMLType::F32)
                        .with_context(|| format!("tensor {hf_name}: cast failed"))?;
                    let gs = py_shape.iter().rev().map(|&d| d as u64).collect();
                    (GGMLType::F32, gs, data)
                } else {
                    let dst = opts.output_dtype;
                    let data = cast_data(raw_data, src_dtype, dst)
                        .with_context(|| format!("tensor {hf_name}: cast failed"))?;
                    let gs = py_shape.iter().rev().map(|&d| d as u64).collect();
                    (dst, gs, data)
                };

            writer.add_tensor(gguf_name, gguf_shape, dst_dtype, tensor_data);
            mapped += 1;
            pb.inc(1);
        }
    }

    pb.finish_with_message("tensors processed");

    if !skipped.is_empty() {
        eprintln!("\nWarning: {} tensor(s) skipped (unrecognised names):", skipped.len());
        for name in &skipped {
            eprintln!("  {name}");
        }
    }
    if fallback_count > 0 {
        eprintln!("\nNote: {fallback_count} tensor(s) fell back to F32 (too small for Q8_0 blocks).");
    }

    println!("Writing {mapped} tensors to {} …", output_path.display());
    let out_file = File::create(output_path)
        .with_context(|| format!("create output file {}", output_path.display()))?;
    let mut buf_writer = BufWriter::new(out_file);
    writer.write_to(&mut buf_writer)?;
    println!("Done.");

    Ok(())
}

fn write_metadata(writer: &mut GGUFWriter, model_id: &str, config: &TtmConfig) {
    writer.add_metadata("general.architecture",    GGUFMetaValue::String("ttm".into()));
    writer.add_metadata("general.name",            GGUFMetaValue::String(model_id.into()));
    writer.add_metadata("ttm.context_length",      GGUFMetaValue::Uint32(config.context_length as u32));
    writer.add_metadata("ttm.prediction_length",   GGUFMetaValue::Uint32(config.prediction_length as u32));
    writer.add_metadata("ttm.patch_length",        GGUFMetaValue::Uint32(config.patch_length as u32));
    writer.add_metadata("ttm.patch_stride",        GGUFMetaValue::Uint32(config.patch_stride as u32));
    writer.add_metadata("ttm.d_model",             GGUFMetaValue::Uint32(config.d_model as u32));
    writer.add_metadata("ttm.num_layers",          GGUFMetaValue::Uint32(config.num_layers as u32));
    writer.add_metadata("ttm.decoder_d_model",     GGUFMetaValue::Uint32(config.decoder_d_model as u32));
    writer.add_metadata("ttm.decoder_num_layers",  GGUFMetaValue::Uint32(config.decoder_num_layers as u32));
    writer.add_metadata("ttm.expansion_factor",    GGUFMetaValue::Uint32(config.expansion_factor as u32));
    writer.add_metadata("ttm.adaptive_patching_levels", GGUFMetaValue::Uint32(config.adaptive_patching_levels as u32));
    writer.add_metadata("ttm.num_patches",         GGUFMetaValue::Uint32(config.num_patches as u32));
    writer.add_metadata("ttm.scaling",             GGUFMetaValue::String(config.scaling.clone()));
    writer.add_metadata("ttm.gated_attn",          GGUFMetaValue::Bool(config.gated_attn));
    writer.add_metadata("ttm.norm_eps",            GGUFMetaValue::Float64(config.norm_eps));
}

fn ggml_type_from_st(dtype: StDtype) -> anyhow::Result<GGMLType> {
    match dtype {
        StDtype::F32  => Ok(GGMLType::F32),
        StDtype::F16  => Ok(GGMLType::F16),
        StDtype::BF16 => Ok(GGMLType::BF16),
        other => anyhow::bail!("unsupported safetensors dtype: {other:?}"),
    }
}

fn load_shard_bytes(shards: &[std::path::PathBuf]) -> anyhow::Result<Vec<Vec<u8>>> {
    shards
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("read shard {}", p.display())))
        .collect()
}

fn cast_data(data: &[u8], src: GGMLType, dst: GGMLType) -> anyhow::Result<Vec<u8>> {
    if src == dst {
        return Ok(data.to_vec());
    }
    if dst == GGMLType::Q8_0 {
        let f32_values = decode_to_f32(data, src)?;
        return quantize_q8_0(&f32_values);
    }
    match (src, dst) {
        (GGMLType::F32, GGMLType::F16) => {
            let f32_values = parse_f32_le(data)?;
            let mut out = Vec::with_capacity(f32_values.len() * 2);
            for v in f32_values {
                let bits = f32_to_f16_bits(v);
                out.extend_from_slice(&bits.to_le_bytes());
            }
            Ok(out)
        }
        (GGMLType::BF16, GGMLType::F32) => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for chunk in data.chunks_exact(2) {
                let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_bits = (bf16_bits as u32) << 16;
                out.extend_from_slice(&f32_bits.to_le_bytes());
            }
            Ok(out)
        }
        (GGMLType::BF16, GGMLType::F16) => {
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_bits = (bf16_bits as u32) << 16;
                let f32_val = f32::from_bits(f32_bits);
                let f16_bits = f32_to_f16_bits(f32_val);
                out.extend_from_slice(&f16_bits.to_le_bytes());
            }
            Ok(out)
        }
        (GGMLType::F16, GGMLType::F32) => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for chunk in data.chunks_exact(2) {
                let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f32_val = f16_to_f32(f16_bits);
                out.extend_from_slice(&f32_val.to_bits().to_le_bytes());
            }
            Ok(out)
        }
        _ => anyhow::bail!("unsupported cast: {src:?} → {dst:?}"),
    }
}

fn decode_to_f32(data: &[u8], src: GGMLType) -> anyhow::Result<Vec<f32>> {
    match src {
        GGMLType::F32  => parse_f32_le(data),
        GGMLType::F16  => data
            .chunks_exact(2)
            .map(|c| Ok(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))))
            .collect(),
        GGMLType::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let bf16_bits = u16::from_le_bytes([c[0], c[1]]);
                Ok(f32::from_bits((bf16_bits as u32) << 16))
            })
            .collect(),
        GGMLType::Q8_0 => anyhow::bail!("Q8_0 → re-quantization not supported as source"),
    }
}

fn transpose_f32(data: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            out[c * n_rows + r] = data[r * n_cols + c];
        }
    }
    out
}

fn quantize_q8_0(values: &[f32]) -> anyhow::Result<Vec<u8>> {
    const BLOCK: usize = 32;
    if values.len() % BLOCK != 0 {
        anyhow::bail!("Q8_0 requires element count divisible by {BLOCK}, got {}", values.len());
    }
    let n_blocks = values.len() / BLOCK;
    let mut out = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let blk = &values[b * BLOCK..(b + 1) * BLOCK];
        let amax = blk.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let d = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
        let d_inv = if d == 0.0 { 0.0f32 } else { 1.0 / d };
        let base = b * 34;
        let d_f16 = f32_to_f16_bits(d);
        out[base..base + 2].copy_from_slice(&d_f16.to_le_bytes());
        for i in 0..BLOCK {
            let q = (blk[i] * d_inv).round().clamp(-127.0, 127.0) as i8;
            out[base + 2 + i] = q as u8;
        }
    }
    Ok(out)
}

fn parse_f32_le(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    if data.len() % 4 != 0 {
        anyhow::bail!("f32 data length not divisible by 4");
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return sign | 0x7C00; }
    if new_exp <= 0 {
        if new_exp < -10 { return sign; }
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp);
        return sign | (m >> 13) as u16;
    }
    sign | ((new_exp as u16) << 10) | (mantissa >> 13) as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as u32;
    let f32_bits = if exp == 0 {
        if mantissa == 0 { sign }
        else {
            let mut m = mantissa;
            let mut e = 0i32;
            while m & 0x0400 == 0 { m <<= 1; e += 1; }
            sign | ((127 - 15 - e + 1) as u32) << 23 | (m & 0x03FF) << 13
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 127 - 15) as u32) << 23 | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}
