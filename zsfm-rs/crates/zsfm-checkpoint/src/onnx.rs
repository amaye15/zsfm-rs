//! ONNX checkpoint loading: weights are the graph's initializer tensors.
//!
//! ONNX is protobuf, but the weight layout only needs a handful of fields
//! (ModelProto.graph → GraphProto.initializer → TensorProto), so this module
//! decodes the protobuf wire format directly — no protoc / prost build-time
//! dependency. Both storage layouts are handled: `raw_data` bytes and the typed
//! repeated fields (`float_data`, `int32_data`, …), packed or unpacked.
//! F32/F16/BF16 are preserved exactly; DOUBLE and integer/bool types are cast
//! to F32 with a note. Models using external data files are rejected with
//! guidance.

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::cast::{f32_to_bytes, SrcDtype};
use crate::read::{Checkpoint, RawTensor};

// onnx.TensorProto.DataType values.
const DT_FLOAT: i32 = 1;
const DT_UINT8: i32 = 2;
const DT_INT8: i32 = 3;
const DT_UINT16: i32 = 4;
const DT_INT16: i32 = 5;
const DT_INT32: i32 = 6;
const DT_INT64: i32 = 7;
const DT_BOOL: i32 = 9;
const DT_FLOAT16: i32 = 10;
const DT_DOUBLE: i32 = 11;
const DT_UINT32: i32 = 12;
const DT_UINT64: i32 = 13;
const DT_BFLOAT16: i32 = 16;

const LOCATION_EXTERNAL: u64 = 1;

pub fn load_onnx(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let initializers = parse_model(&bytes).context("parse ONNX protobuf")?;
    anyhow::ensure!(
        !initializers.is_empty(),
        "ONNX graph has no initializer tensors — the weights may live in external \
         data files or be supplied as runtime inputs"
    );

    let mut tensors = Vec::with_capacity(initializers.len());
    for t in &initializers {
        tensors.push(tensor_proto_to_raw(t)?);
    }
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format reader
// ---------------------------------------------------------------------------

struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .context("protobuf: truncated varint")?;
            self.pos += 1;
            if shift < 64 {
                value |= u64::from(byte & 0x7F) << shift;
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            anyhow::ensure!(shift <= 70, "protobuf: varint too long");
        }
    }

    /// Returns (field_number, wire_type).
    fn read_tag(&mut self) -> Result<(u64, u8)> {
        let tag = self.read_varint()?;
        Ok((tag >> 3, (tag & 0x7) as u8))
    }

    fn read_len_delimited(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.buf.len())
            .context("protobuf: truncated length-delimited field")?;
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_fixed32(&mut self) -> Result<u32> {
        let end = self.pos + 4;
        anyhow::ensure!(end <= self.buf.len(), "protobuf: truncated fixed32");
        let v = u32::from_le_bytes(self.buf[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn read_fixed64(&mut self) -> Result<u64> {
        let end = self.pos + 8;
        anyhow::ensure!(end <= self.buf.len(), "protobuf: truncated fixed64");
        let v = u64::from_le_bytes(self.buf[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn skip(&mut self, wire_type: u8) -> Result<()> {
        match wire_type {
            0 => {
                self.read_varint()?;
            }
            1 => {
                self.read_fixed64()?;
            }
            2 => {
                self.read_len_delimited()?;
            }
            5 => {
                self.read_fixed32()?;
            }
            other => bail!("protobuf: unsupported wire type {other}"),
        }
        Ok(())
    }
}

#[derive(Default)]
struct TensorProto {
    dims: Vec<i64>,
    data_type: i32,
    float_data: Vec<f32>,
    int32_data: Vec<i32>,
    int64_data: Vec<i64>,
    uint64_data: Vec<u64>,
    double_data: Vec<f64>,
    name: String,
    raw_data: Vec<u8>,
    data_location: u64,
}

/// ModelProto: field 7 = graph (GraphProto).
fn parse_model(buf: &[u8]) -> Result<Vec<TensorProto>> {
    let mut r = ProtoReader::new(buf);
    let mut initializers = Vec::new();
    while !r.done() {
        let (field, wire) = r.read_tag()?;
        if field == 7 && wire == 2 {
            let graph = r.read_len_delimited()?;
            initializers.extend(parse_graph(graph)?);
        } else {
            r.skip(wire)?;
        }
    }
    Ok(initializers)
}

/// GraphProto: field 5 = initializer (repeated TensorProto).
fn parse_graph(buf: &[u8]) -> Result<Vec<TensorProto>> {
    let mut r = ProtoReader::new(buf);
    let mut initializers = Vec::new();
    while !r.done() {
        let (field, wire) = r.read_tag()?;
        if field == 5 && wire == 2 {
            let tensor = r.read_len_delimited()?;
            initializers.push(parse_tensor(tensor)?);
        } else {
            r.skip(wire)?;
        }
    }
    Ok(initializers)
}

fn parse_tensor(buf: &[u8]) -> Result<TensorProto> {
    let mut r = ProtoReader::new(buf);
    let mut t = TensorProto::default();
    while !r.done() {
        let (field, wire) = r.read_tag()?;
        match (field, wire) {
            // dims: repeated int64 (packed or unpacked)
            (1, 0) => t.dims.push(r.read_varint()? as i64),
            (1, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.dims.push(p.read_varint()? as i64);
                }
            }
            (2, 0) => t.data_type = r.read_varint()? as i32,
            // float_data: repeated float (packed or unpacked)
            (4, 5) => t.float_data.push(f32::from_bits(r.read_fixed32()?)),
            (4, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.float_data.push(f32::from_bits(p.read_fixed32()?));
                }
            }
            // int32_data: repeated int32
            (5, 0) => t.int32_data.push(r.read_varint()? as i64 as i32),
            (5, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.int32_data.push(p.read_varint()? as i64 as i32);
                }
            }
            // int64_data: repeated int64
            (7, 0) => t.int64_data.push(r.read_varint()? as i64),
            (7, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.int64_data.push(p.read_varint()? as i64);
                }
            }
            (8, 2) => {
                t.name = String::from_utf8_lossy(r.read_len_delimited()?).into_owned();
            }
            (9, 2) => t.raw_data = r.read_len_delimited()?.to_vec(),
            // double_data: repeated double
            (10, 1) => t.double_data.push(f64::from_bits(r.read_fixed64()?)),
            (10, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.double_data.push(f64::from_bits(p.read_fixed64()?));
                }
            }
            // uint64_data: repeated uint64
            (11, 0) => t.uint64_data.push(r.read_varint()?),
            (11, 2) => {
                let mut p = ProtoReader::new(r.read_len_delimited()?);
                while !p.done() {
                    t.uint64_data.push(p.read_varint()?);
                }
            }
            (14, 0) => t.data_location = r.read_varint()?,
            (_, w) => r.skip(w)?,
        }
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// TensorProto → RawTensor
// ---------------------------------------------------------------------------

fn tensor_proto_to_raw(t: &TensorProto) -> Result<RawTensor> {
    let name = t.name.clone();
    if t.data_location == LOCATION_EXTERNAL {
        bail!(
            "tensor {name}: stored as external data — re-export the model with weights \
             embedded (onnx.external_data_helper.load_external_data_for_model + save) \
             and convert again"
        );
    }
    let shape: Vec<u64> = t.dims.iter().map(|&d| d as u64).collect();

    let (dtype, data): (SrcDtype, Vec<u8>) = match t.data_type {
        DT_FLOAT => {
            if !t.raw_data.is_empty() {
                (SrcDtype::F32, t.raw_data.clone())
            } else {
                (SrcDtype::F32, f32_to_bytes(&t.float_data))
            }
        }
        DT_FLOAT16 | DT_BFLOAT16 => {
            let dtype = if t.data_type == DT_FLOAT16 { SrcDtype::F16 } else { SrcDtype::BF16 };
            if !t.raw_data.is_empty() {
                (dtype, t.raw_data.clone())
            } else {
                // Spec: 16-bit floats without raw_data are stored as the low 16
                // bits of int32_data entries.
                let bytes = t
                    .int32_data
                    .iter()
                    .flat_map(|&v| (v as u16).to_le_bytes())
                    .collect();
                (dtype, bytes)
            }
        }
        DT_DOUBLE => {
            eprintln!("note: tensor {name}: casting DOUBLE to F32");
            let vals: Vec<f32> = if !t.raw_data.is_empty() {
                t.raw_data
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            } else {
                t.double_data.iter().map(|&v| v as f32).collect()
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        DT_INT64 => {
            eprintln!("note: tensor {name}: casting INT64 to F32");
            let vals: Vec<f32> = if !t.raw_data.is_empty() {
                t.raw_data
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            } else {
                t.int64_data.iter().map(|&v| v as f32).collect()
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        DT_UINT64 => {
            eprintln!("note: tensor {name}: casting UINT64 to F32");
            let vals: Vec<f32> = if !t.raw_data.is_empty() {
                t.raw_data
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect()
            } else {
                t.uint64_data.iter().map(|&v| v as f32).collect()
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        DT_INT32 | DT_UINT32 | DT_INT16 | DT_UINT16 | DT_INT8 | DT_UINT8 | DT_BOOL => {
            eprintln!("note: tensor {name}: casting integer/bool data to F32");
            let vals: Vec<f32> = if !t.raw_data.is_empty() {
                decode_small_ints(&t.raw_data, t.data_type)?
            } else {
                // The typed field for every sub-32-bit integer type is int32_data.
                t.int32_data.iter().map(|&v| v as f32).collect()
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        other => bail!("tensor {name}: ONNX data_type {other} not supported"),
    };

    Ok(RawTensor { name, shape, dtype, data })
}

fn decode_small_ints(raw: &[u8], data_type: i32) -> Result<Vec<f32>> {
    Ok(match data_type {
        DT_INT32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DT_UINT32 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DT_INT16 => raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DT_UINT16 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DT_INT8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        DT_UINT8 => raw.iter().map(|&b| b as f32).collect(),
        DT_BOOL => raw.iter().map(|&b| (b != 0) as u8 as f32).collect(),
        other => bail!("unexpected small-int data_type {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-encoded protobuf helpers for building fixture bytes.
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }

    fn tag(field: u64, wire: u8, out: &mut Vec<u8>) {
        varint(field << 3 | wire as u64, out);
    }

    fn len_delim(field: u64, payload: &[u8], out: &mut Vec<u8>) {
        tag(field, 2, out);
        varint(payload.len() as u64, out);
        out.extend_from_slice(payload);
    }

    fn build_model(tensor: &[u8]) -> Vec<u8> {
        let mut graph = Vec::new();
        len_delim(5, tensor, &mut graph); // GraphProto.initializer
        let mut model = Vec::new();
        tag(1, 0, &mut model); // ModelProto.ir_version
        varint(8, &mut model);
        len_delim(7, &graph, &mut model); // ModelProto.graph
        model
    }

    #[test]
    fn parses_f32_raw_data_tensor() {
        let vals = [1.5f32, -2.25, 3.0];
        let mut t = Vec::new();
        // dims: packed [3]
        let mut dims = Vec::new();
        varint(3, &mut dims);
        len_delim(1, &dims, &mut t);
        tag(2, 0, &mut t); // data_type = FLOAT
        varint(DT_FLOAT as u64, &mut t);
        len_delim(8, b"w", &mut t); // name
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        len_delim(9, &raw, &mut t); // raw_data

        let model = build_model(&t);
        let init = parse_model(&model).unwrap();
        assert_eq!(init.len(), 1);
        let rt = tensor_proto_to_raw(&init[0]).unwrap();
        assert_eq!(rt.name, "w");
        assert_eq!(rt.shape, vec![3]);
        assert_eq!(rt.dtype, SrcDtype::F32);
        let got: Vec<f32> = rt
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, vals);
    }

    #[test]
    fn parses_packed_float_data_and_int64_dims_unpacked() {
        let mut t = Vec::new();
        tag(1, 0, &mut t); // dims unpacked: 2
        varint(2, &mut t);
        tag(1, 0, &mut t); // dims unpacked: 2
        varint(2, &mut t);
        tag(2, 0, &mut t);
        varint(DT_FLOAT as u64, &mut t);
        len_delim(8, b"packed", &mut t);
        let mut fd = Vec::new();
        for v in [0.5f32, 1.0, 1.5, 2.0] {
            fd.extend_from_slice(&v.to_le_bytes());
        }
        len_delim(4, &fd, &mut t); // float_data packed

        let init = parse_model(&build_model(&t)).unwrap();
        let rt = tensor_proto_to_raw(&init[0]).unwrap();
        assert_eq!(rt.shape, vec![2, 2]);
        assert_eq!(rt.data.len(), 16);
    }

    #[test]
    fn rejects_external_data() {
        let mut t = Vec::new();
        tag(2, 0, &mut t);
        varint(DT_FLOAT as u64, &mut t);
        len_delim(8, b"ext", &mut t);
        tag(14, 0, &mut t); // data_location = EXTERNAL
        varint(LOCATION_EXTERNAL, &mut t);

        let init = parse_model(&build_model(&t)).unwrap();
        let err = match tensor_proto_to_raw(&init[0]) {
            Ok(_) => panic!("expected external-data error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("external data"));
    }
}
