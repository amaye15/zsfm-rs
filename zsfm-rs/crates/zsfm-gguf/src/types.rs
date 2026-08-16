/// GGML tensor data types used in GGUF files.
/// Values match the ggml_type enum in ggml.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    /// Q8_0: blocks of 32 × i8 with a shared f16 scale (34 bytes/block).
    Q8_0 = 8,
    BF16 = 30,
}

impl GGMLType {
    /// (block_elems, bytes_per_block) for block-quantized types; None for float types.
    /// Q8_0: 32 × i8 values + 1 × f16 scale = 34 bytes/block.
    #[allow(dead_code)]
    pub fn block_shape(self) -> Option<(usize, usize)> {
        match self {
            GGMLType::Q8_0 => Some((32, 34)),
            _ => None,
        }
    }
}

/// GGUF metadata value types (gguf_metadata_value_type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(dead_code)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// A typed metadata value for a GGUF key-value pair.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum GGUFMetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    ArrayUint32(Vec<u32>),
    ArrayFloat32(Vec<f32>),
    ArrayString(Vec<String>),
}

impl GGUFMetaValue {
    pub fn value_type(&self) -> GGUFValueType {
        match self {
            GGUFMetaValue::Uint8(_) => GGUFValueType::Uint8,
            GGUFMetaValue::Int8(_) => GGUFValueType::Int8,
            GGUFMetaValue::Uint16(_) => GGUFValueType::Uint16,
            GGUFMetaValue::Int16(_) => GGUFValueType::Int16,
            GGUFMetaValue::Uint32(_) => GGUFValueType::Uint32,
            GGUFMetaValue::Int32(_) => GGUFValueType::Int32,
            GGUFMetaValue::Float32(_) => GGUFValueType::Float32,
            GGUFMetaValue::Bool(_) => GGUFValueType::Bool,
            GGUFMetaValue::String(_) => GGUFValueType::String,
            GGUFMetaValue::Uint64(_) => GGUFValueType::Uint64,
            GGUFMetaValue::Int64(_) => GGUFValueType::Int64,
            GGUFMetaValue::Float64(_) => GGUFValueType::Float64,
            GGUFMetaValue::ArrayUint32(_)
            | GGUFMetaValue::ArrayFloat32(_)
            | GGUFMetaValue::ArrayString(_) => GGUFValueType::Array,
        }
    }
}
