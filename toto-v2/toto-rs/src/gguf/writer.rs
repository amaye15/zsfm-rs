use std::io::{self, Seek, Write};

use byteorder::{LittleEndian, WriteBytesExt};

use super::types::{GGMLType, GGUFMetaValue, GGUFValueType};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const ALIGNMENT: u64 = 32;

/// Describes one tensor's position in the GGUF data section.
#[derive(Debug)]
struct TensorInfo {
    name: String,
    shape: Vec<u64>,
    dtype: GGMLType,
    /// Byte offset from the start of the tensor data block.
    offset: u64,
    /// Raw tensor bytes (row-major, little-endian).
    data: Vec<u8>,
}

/// Streaming GGUF v3 writer.
///
/// Call [`add_metadata`] for every key-value pair, then [`add_tensor`] for
/// every tensor, then [`write_to`] to flush the complete file.
pub struct GGUFWriter {
    metadata: Vec<(String, GGUFMetaValue)>,
    tensors: Vec<TensorInfo>,
    /// Running data-block offset for the next tensor.
    data_offset: u64,
}

impl GGUFWriter {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            data_offset: 0,
        }
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: GGUFMetaValue) {
        self.metadata.push((key.into(), value));
    }

    /// Buffer a tensor. `data` must already be in the target dtype byte layout.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<u64>,
        dtype: GGMLType,
        data: Vec<u8>,
    ) {
        let offset = self.data_offset;
        let padded_len = round_up(data.len() as u64, ALIGNMENT);
        self.data_offset += padded_len;
        self.tensors.push(TensorInfo {
            name: name.into(),
            shape,
            dtype,
            offset,
            data,
        });
    }

    /// Serialize the complete GGUF file to `writer`.
    pub fn write_to<W: Write + Seek>(&self, writer: &mut W) -> anyhow::Result<()> {
        // --- header ---
        writer.write_all(GGUF_MAGIC)?;
        writer.write_u32::<LittleEndian>(GGUF_VERSION)?;
        writer.write_u64::<LittleEndian>(self.tensors.len() as u64)?;
        writer.write_u64::<LittleEndian>(self.metadata.len() as u64)?;

        // --- metadata key-value pairs ---
        for (key, value) in &self.metadata {
            write_string(writer, key)?;
            writer.write_u32::<LittleEndian>(value.value_type() as u32)?;
            write_value(writer, value)?;
        }

        // --- tensor info ---
        for t in &self.tensors {
            write_string(writer, &t.name)?;
            writer.write_u32::<LittleEndian>(t.shape.len() as u32)?;
            for &dim in &t.shape {
                writer.write_u64::<LittleEndian>(dim)?;
            }
            writer.write_u32::<LittleEndian>(t.dtype as u32)?;
            writer.write_u64::<LittleEndian>(t.offset)?;
        }

        // --- align to ALIGNMENT before tensor data ---
        let pos = writer.stream_position()?;
        let aligned = round_up(pos, ALIGNMENT);
        if aligned > pos {
            let pad = vec![0u8; (aligned - pos) as usize];
            writer.write_all(&pad)?;
        }

        // --- tensor data (each padded to ALIGNMENT) ---
        for t in &self.tensors {
            writer.write_all(&t.data)?;
            let remainder = t.data.len() as u64 % ALIGNMENT;
            if remainder != 0 {
                let pad = vec![0u8; (ALIGNMENT - remainder) as usize];
                writer.write_all(&pad)?;
            }
        }

        Ok(())
    }
}

fn round_up(value: u64, align: u64) -> u64 {
    (value + align - 1) / align * align
}

fn write_string<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    writer.write_u64::<LittleEndian>(s.len() as u64)?;
    writer.write_all(s.as_bytes())
}

fn write_value<W: Write>(writer: &mut W, value: &GGUFMetaValue) -> anyhow::Result<()> {
    match value {
        GGUFMetaValue::Uint8(v) => writer.write_u8(*v)?,
        GGUFMetaValue::Int8(v) => writer.write_i8(*v)?,
        GGUFMetaValue::Uint16(v) => writer.write_u16::<LittleEndian>(*v)?,
        GGUFMetaValue::Int16(v) => writer.write_i16::<LittleEndian>(*v)?,
        GGUFMetaValue::Uint32(v) => writer.write_u32::<LittleEndian>(*v)?,
        GGUFMetaValue::Int32(v) => writer.write_i32::<LittleEndian>(*v)?,
        GGUFMetaValue::Float32(v) => writer.write_f32::<LittleEndian>(*v)?,
        GGUFMetaValue::Bool(v) => writer.write_u8(*v as u8)?,
        GGUFMetaValue::String(v) => write_string(writer, v)?,
        GGUFMetaValue::Uint64(v) => writer.write_u64::<LittleEndian>(*v)?,
        GGUFMetaValue::Int64(v) => writer.write_i64::<LittleEndian>(*v)?,
        GGUFMetaValue::Float64(v) => writer.write_f64::<LittleEndian>(*v)?,
        GGUFMetaValue::ArrayUint32(arr) => {
            writer.write_u32::<LittleEndian>(GGUFValueType::Uint32 as u32)?;
            writer.write_u64::<LittleEndian>(arr.len() as u64)?;
            for v in arr {
                writer.write_u32::<LittleEndian>(*v)?;
            }
        }
        GGUFMetaValue::ArrayString(arr) => {
            writer.write_u32::<LittleEndian>(GGUFValueType::String as u32)?;
            writer.write_u64::<LittleEndian>(arr.len() as u64)?;
            for s in arr {
                write_string(writer, s)?;
            }
        }
    }
    Ok(())
}
