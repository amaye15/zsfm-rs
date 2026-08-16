//! Minimal GGUF v2/v3 reader for the tensor types this crate's writer emits
//! (F32, F16, BF16, Q8_0).
//!
//! candle's `gguf_file` reader covers many more quantization formats but does
//! not know BF16 (ggml dtype 30), so BF16 files written here would otherwise be
//! unreadable by our own tooling. Callers should prefer candle's reader and
//! fall back to this one.

use std::io::{Read, Seek, SeekFrom};

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};

use super::types::{GGMLType, GGUFMetaValue};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

/// Capacity every model crate should wrap its `BufReader<File>` with before parsing a GGUF
/// file. Header/metadata parsing is hundreds to thousands of few-byte `read_u32`/`read_u64`
/// calls (one per tensor dim, name length, metadata value, …); the default 8 KiB `BufReader`
/// capacity is fine for small files but forces a syscall every few tensors on models with
/// hundreds of them. 1 MiB comfortably covers any GGUF header in one underlying read while
/// staying small relative to the model file itself.
pub const READ_BUF_CAPACITY: usize = 1 << 20;

pub struct GGUFTensorInfo {
    pub name: String,
    /// Dims exactly as stored in the file (GGUF order — reversed row-major).
    pub shape: Vec<u64>,
    pub dtype: GGMLType,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
}

impl GGUFTensorInfo {
    pub fn n_elems(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn n_bytes(&self) -> u64 {
        match self.dtype {
            GGMLType::F32 => self.n_elems() * 4,
            GGMLType::F16 | GGMLType::BF16 => self.n_elems() * 2,
            GGMLType::Q8_0 => self.n_elems() / 32 * 34,
        }
    }
}

pub struct GGUFFile {
    pub metadata: Vec<(String, GGUFMetaValue)>,
    pub tensors: Vec<GGUFTensorInfo>,
    /// Absolute file offset of the tensor data section.
    pub data_start: u64,
}

impl GGUFFile {
    pub fn read(reader: &mut (impl Read + Seek)) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).context("read GGUF magic")?;
        if &magic != GGUF_MAGIC {
            bail!("not a GGUF file (bad magic)");
        }
        let version = reader.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let kv_count = reader.read_u64::<LittleEndian>()?;

        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(reader)?;
            let vtype = reader.read_u32::<LittleEndian>()?;
            let value = read_value(reader, vtype)?;
            if let Some(v) = value {
                metadata.push((key, v));
            }
        }

        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                GGUFMetaValue::Uint32(a) => Some(*a as u64),
                GGUFMetaValue::Uint64(a) => Some(*a),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(reader)?;
            let n_dims = reader.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(reader.read_u64::<LittleEndian>()?);
            }
            let dtype_raw = reader.read_u32::<LittleEndian>()?;
            let dtype = match dtype_raw {
                0 => GGMLType::F32,
                1 => GGMLType::F16,
                8 => GGMLType::Q8_0,
                30 => GGMLType::BF16,
                other => bail!("tensor {name}: ggml dtype {other} not supported by this reader"),
            };
            let offset = reader.read_u64::<LittleEndian>()?;
            tensors.push(GGUFTensorInfo { name, shape, dtype, offset });
        }

        let pos = reader.stream_position()?;
        let data_start = pos.div_ceil(alignment) * alignment;

        Ok(GGUFFile { metadata, tensors, data_start })
    }

    /// Raw stored bytes of one tensor.
    pub fn tensor_bytes(
        &self,
        reader: &mut (impl Read + Seek),
        info: &GGUFTensorInfo,
    ) -> Result<Vec<u8>> {
        reader.seek(SeekFrom::Start(self.data_start + info.offset))?;
        let mut buf = vec![0u8; info.n_bytes() as usize];
        reader
            .read_exact(&mut buf)
            .with_context(|| format!("read tensor {} data", info.name))?;
        Ok(buf)
    }

    /// Tensor decoded to f32 (dequantizing Q8_0).
    pub fn tensor_f32(
        &self,
        reader: &mut (impl Read + Seek),
        info: &GGUFTensorInfo,
    ) -> Result<Vec<f32>> {
        let bytes = self.tensor_bytes(reader, info)?;
        match info.dtype {
            GGMLType::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()),
            GGMLType::F16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            GGMLType::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()),
            GGMLType::Q8_0 => {
                let mut out = Vec::with_capacity(info.n_elems() as usize);
                for block in bytes.chunks_exact(34) {
                    let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    for &q in &block[2..34] {
                        out.push(d * (q as i8) as f32);
                    }
                }
                Ok(out)
            }
        }
    }
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).context("GGUF string is not UTF-8")
}

/// Parse one metadata value. Returns `None` for value types we don't model
/// (after consuming the correct number of bytes so the stream stays aligned).
fn read_value(reader: &mut impl Read, vtype: u32) -> Result<Option<GGUFMetaValue>> {
    Ok(match vtype {
        0 => Some(GGUFMetaValue::Uint8(reader.read_u8()?)),
        1 => Some(GGUFMetaValue::Int8(reader.read_i8()?)),
        2 => Some(GGUFMetaValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        3 => Some(GGUFMetaValue::Int16(reader.read_i16::<LittleEndian>()?)),
        4 => Some(GGUFMetaValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        5 => Some(GGUFMetaValue::Int32(reader.read_i32::<LittleEndian>()?)),
        6 => Some(GGUFMetaValue::Float32(reader.read_f32::<LittleEndian>()?)),
        7 => Some(GGUFMetaValue::Bool(reader.read_u8()? != 0)),
        8 => Some(GGUFMetaValue::String(read_string(reader)?)),
        9 => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()?;
            match elem_type {
                4 => {
                    let mut v = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        v.push(reader.read_u32::<LittleEndian>()?);
                    }
                    Some(GGUFMetaValue::ArrayUint32(v))
                }
                6 => {
                    let mut v = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        v.push(reader.read_f32::<LittleEndian>()?);
                    }
                    Some(GGUFMetaValue::ArrayFloat32(v))
                }
                8 => {
                    let mut v = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        v.push(read_string(reader)?);
                    }
                    Some(GGUFMetaValue::ArrayString(v))
                }
                other => {
                    for _ in 0..count {
                        read_value(reader, other)?;
                    }
                    None
                }
            }
        }
        10 => Some(GGUFMetaValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        11 => Some(GGUFMetaValue::Int64(reader.read_i64::<LittleEndian>()?)),
        12 => Some(GGUFMetaValue::Float64(reader.read_f64::<LittleEndian>()?)),
        other => bail!("unsupported GGUF metadata value type {other}"),
    })
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x03FF) as u32;
    let f32_bits = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut m = mantissa;
            let mut e = 0i32;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | ((127 - 15 - e + 1) as u32) << 23 | (m & 0x03FF) << 13
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 127 - 15) as u32) << 23 | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::GGUFWriter;
    use std::io::Cursor;

    #[test]
    fn writer_reader_roundtrip_all_dtypes() {
        let mut w = GGUFWriter::new();
        w.add_metadata("general.architecture", GGUFMetaValue::String("test".into()));
        w.add_metadata("test.count", GGUFMetaValue::Uint32(7));
        w.add_metadata(
            "test.floats",
            GGUFMetaValue::ArrayFloat32(vec![0.5, 1.5]),
        );

        let f32_vals: Vec<f32> = (0..32).map(|i| i as f32 / 4.0).collect();
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor("t.f32", vec![32], GGMLType::F32, f32_bytes);

        // 1.25 is exactly representable in bf16.
        let bf16_bytes: Vec<u8> = (0..16)
            .flat_map(|_| ((1.25f32.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        w.add_tensor("t.bf16", vec![4, 4], GGMLType::BF16, bf16_bytes);

        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();

        buf.set_position(0);
        let f = GGUFFile::read(&mut buf).unwrap();
        assert_eq!(f.metadata.len(), 3);
        assert_eq!(f.tensors.len(), 2);

        let tb = f.tensors.iter().find(|t| t.name == "t.bf16").unwrap();
        assert_eq!(tb.dtype, GGMLType::BF16);
        assert_eq!(tb.shape, vec![4, 4]);
        let vals = f.tensor_f32(&mut buf, tb).unwrap();
        assert!(vals.iter().all(|&v| v == 1.25));

        let tf = f.tensors.iter().find(|t| t.name == "t.f32").unwrap();
        let vals = f.tensor_f32(&mut buf, tf).unwrap();
        assert_eq!(vals, f32_vals);
    }

    #[test]
    fn q8_0_dequant_roundtrip() {
        // d = 1.0, quants = -127..-96 → values exactly d*q.
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // f16 1.0
        for i in 0..32 {
            block[2 + i] = (-(127 - i as i8)) as u8;
        }
        let mut w = GGUFWriter::new();
        w.add_tensor("q", vec![32], GGMLType::Q8_0, block);
        let mut buf = Cursor::new(Vec::new());
        w.write_to(&mut buf).unwrap();
        buf.set_position(0);
        let f = GGUFFile::read(&mut buf).unwrap();
        let vals = f.tensor_f32(&mut buf, &f.tensors[0]).unwrap();
        assert_eq!(vals[0], -127.0);
        assert_eq!(vals[31], -96.0);
    }
}
