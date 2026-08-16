use std::io::{Read, Seek};

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

/// Dequantize a named GGUF tensor and cast it to `dtype`.
pub fn load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let qt = content
        .tensor(reader, name, device)
        .with_context(|| format!("tensor '{name}' not found in GGUF"))?;
    Ok(qt.dequantize(device)?.to_dtype(dtype)?)
}

/// Same as [`load_tensor`], but returns `Ok(None)` instead of erroring when the tensor is absent.
pub fn try_load_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    match content.tensor(reader, name, device) {
        Ok(qt) => Ok(Some(qt.dequantize(device)?.to_dtype(dtype)?)),
        Err(_) => Ok(None),
    }
}

/// Load an F32 weight PyTorch stores as `(d_out, d_in)`. Candle reverses the GGUF shape back to
/// `(d_out, d_in)`; the `dim(0)` check guards against a stray transposed store (the Q8_0
/// transpose trick some converters apply) by flipping the tensor back.
pub fn load_weight(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    expected_d_out: usize,
    device: &Device,
) -> Result<Tensor> {
    let w = load_tensor(content, reader, name, device, DType::F32)?;
    if w.dim(0)? != expected_d_out {
        Ok(w.t()?.contiguous()?)
    } else {
        Ok(w)
    }
}

/// Dequantize a named GGUF tensor to F32 and flatten it to a plain `Vec<f32>`.
pub fn load_vec(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    name: &str,
    device: &Device,
) -> Result<Vec<f32>> {
    Ok(load_tensor(content, reader, name, device, DType::F32)?
        .flatten_all()?
        .to_vec1()?)
}
