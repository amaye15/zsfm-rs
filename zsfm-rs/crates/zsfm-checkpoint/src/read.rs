//! Format-agnostic checkpoint loading.
//!
//! Supported inputs (detected by extension, with a magic-byte fallback):
//! - `.safetensors` — single file or several shard files
//! - `model.safetensors.index.json` — HF sharded-model index (shards resolved
//!   relative to the index file)
//! - `.pt` / `.pth` / `.bin` / `.ckpt` — PyTorch pickle checkpoints (zip-based
//!   `torch.save` format) via candle's pickle reader
//! - `.npy` / `.npz` — NumPy arrays
//! - `.gguf` — existing GGUF files (tensors dequantized to F32; metadata carried
//!   through), which makes dtype re-quantization possible
//!
//! Every tensor is returned as raw little-endian bytes plus a [`SrcDtype`]
//! (F32/F16/BF16 preserved exactly; other dtypes — F64, integers — are cast to
//! F32 with a warning).

use std::collections::HashSet;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{pickle, DType, Device, Tensor};
use safetensors::{Dtype as StDtype, SafeTensors};

use zsfm_gguf::GGUFMetaValue;

use crate::cast::{f32_to_bytes, SrcDtype};

/// One tensor read from a checkpoint: name, row-major (python-order) shape,
/// source dtype, and raw little-endian bytes.
pub struct RawTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: SrcDtype,
    pub data: Vec<u8>,
}

/// A loaded checkpoint: all tensors plus any metadata carried over from the
/// source (only GGUF inputs have metadata).
pub struct Checkpoint {
    pub tensors: Vec<RawTensor>,
    pub metadata: Vec<(String, GGUFMetaValue)>,
}

#[derive(Default)]
pub struct LoadOptions {
    /// Descend into this top-level dict before collecting tensors (pickle
    /// inputs only), e.g. `state_dict` for PyTorch Lightning checkpoints.
    pub pickle_key: Option<String>,
    /// Strip this prefix from every tensor name that carries it.
    pub strip_prefix: Option<String>,
}

/// Load one or more checkpoint files into a single [`Checkpoint`].
/// Multiple inputs (e.g. safetensors shards) are merged; duplicate tensor
/// names across inputs are an error.
pub fn load_checkpoint(inputs: &[PathBuf], opts: &LoadOptions) -> Result<Checkpoint> {
    anyhow::ensure!(!inputs.is_empty(), "no input files given");

    let mut tensors: Vec<RawTensor> = Vec::new();
    let mut metadata: Vec<(String, GGUFMetaValue)> = Vec::new();

    for path in inputs {
        let ckpt = load_one(path, opts)
            .with_context(|| format!("load checkpoint {}", path.display()))?;
        tensors.extend(ckpt.tensors);
        metadata.extend(ckpt.metadata);
    }

    if let Some(prefix) = &opts.strip_prefix {
        for t in &mut tensors {
            if let Some(rest) = t.name.strip_prefix(prefix.as_str()) {
                t.name = rest.to_string();
            }
        }
    }

    let mut seen = HashSet::new();
    for t in &tensors {
        if !seen.insert(t.name.as_str()) {
            bail!("duplicate tensor name across inputs: {}", t.name);
        }
    }

    Ok(Checkpoint { tensors, metadata })
}

/// TensorFlow SavedModel / TF-checkpoint weights use the TensorBundle format,
/// which has no trustworthy pure-Rust reader — point at the bundled exporter
/// script instead of failing cryptically.
const TF_BUNDLE_HELP: &str =
    "TensorFlow SavedModel / checkpoint weights (TensorBundle format) can't be read \
     natively. Export them to safetensors first with the bundled script (needs \
     tensorflow + safetensors installed):\n\n    \
     python zsfm-rs/tools/export-weights.py <saved_model_dir | ckpt_prefix> weights.safetensors\n\n\
     then run: zsfm convert weights.safetensors -o model.gguf\n\
     (Keras .h5 / .keras files ARE supported natively — no export needed.)";

fn load_one(path: &Path, opts: &LoadOptions) -> Result<Checkpoint> {
    let fname = path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if path.is_dir() {
        if path.join("saved_model.pb").exists() {
            bail!("{} is a TensorFlow SavedModel directory. {TF_BUNDLE_HELP}", path.display());
        }
        bail!(
            "{} is a directory — pass a checkpoint file (or for sharded safetensors, \
             the model.safetensors.index.json)",
            path.display()
        );
    }
    if ext == "index" || fname.contains(".data-00") {
        bail!("{} looks like a TensorFlow checkpoint file. {TF_BUNDLE_HELP}", path.display());
    }

    if fname.ends_with(".safetensors.index.json") {
        return load_safetensors_index(path);
    }
    match ext.as_str() {
        "safetensors" => load_safetensors(path),
        "pt" | "pth" | "bin" | "ckpt" => load_pickle(path, opts),
        "npz" => load_npz(path),
        "npy" => load_npy(path),
        "gguf" => load_gguf(path),
        "onnx" => crate::onnx::load_onnx(path),
        "h5" | "hdf5" => crate::hdf5_keras::load_hdf5(path),
        "keras" => crate::hdf5_keras::load_keras(path),
        _ => match sniff_format(path)? {
            Sniffed::SafeTensors => load_safetensors(path),
            Sniffed::Gguf => load_gguf(path),
            Sniffed::Hdf5 => crate::hdf5_keras::load_hdf5(path),
            Sniffed::Unknown => bail!(
                "unrecognized checkpoint format for {} — supported: .safetensors, \
                 model.safetensors.index.json, .pt/.pth/.bin/.ckpt (PyTorch pickle), \
                 .npy/.npz, .onnx, .h5/.hdf5, .keras, .gguf",
                path.display()
            ),
        },
    }
}

enum Sniffed {
    SafeTensors,
    Gguf,
    Hdf5,
    Unknown,
}

fn sniff_format(path: &Path) -> Result<Sniffed> {
    use std::io::Read;
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read(&mut head)?;
    if n >= 4 && &head[..4] == b"GGUF" {
        return Ok(Sniffed::Gguf);
    }
    if n >= 8 && &head[..8] == b"\x89HDF\r\n\x1a\n" {
        return Ok(Sniffed::Hdf5);
    }
    // safetensors: u64 LE header length followed by a JSON object.
    if n >= 9 {
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        let file_len = std::fs::metadata(path)?.len();
        if header_len > 0 && header_len.saturating_add(8) <= file_len && head[8] == b'{' {
            return Ok(Sniffed::SafeTensors);
        }
    }
    Ok(Sniffed::Unknown)
}

// ---------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------

fn load_safetensors(path: &Path) -> Result<Checkpoint> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("deserialize safetensors")?;

    let mut tensors = Vec::with_capacity(st.len());
    for (name, view) in st.tensors() {
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        let (dtype, data) = match view.dtype() {
            StDtype::F32 => (SrcDtype::F32, view.data().to_vec()),
            StDtype::F16 => (SrcDtype::F16, view.data().to_vec()),
            StDtype::BF16 => (SrcDtype::BF16, view.data().to_vec()),
            other => {
                eprintln!("note: tensor {name}: casting {other:?} to F32");
                let n: usize = view.shape().iter().product();
                let t = safetensors_view_to_f32(&view, other, n)
                    .with_context(|| format!("tensor {name}: unsupported dtype {other:?}"))?;
                (SrcDtype::F32, f32_to_bytes(&t))
            }
        };
        tensors.push(RawTensor { name: name.to_string(), shape, dtype, data });
    }
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

fn safetensors_view_to_f32(
    view: &safetensors::tensor::TensorView,
    dtype: StDtype,
    n_elems: usize,
) -> Result<Vec<f32>> {
    let data = view.data();
    let mut out = Vec::with_capacity(n_elems);
    match dtype {
        StDtype::F64 => {
            for c in data.chunks_exact(8) {
                out.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I64 => {
            for c in data.chunks_exact(8) {
                out.push(i64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I32 => {
            for c in data.chunks_exact(4) {
                out.push(i32::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I16 => {
            for c in data.chunks_exact(2) {
                out.push(i16::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        StDtype::I8 => out.extend(data.iter().map(|&b| b as i8 as f32)),
        StDtype::U8 => out.extend(data.iter().map(|&b| b as f32)),
        StDtype::BOOL => out.extend(data.iter().map(|&b| (b != 0) as u8 as f32)),
        other => bail!("safetensors dtype {other:?} not supported"),
    }
    Ok(out)
}

fn load_safetensors_index(path: &Path) -> Result<Checkpoint> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("parse index json")?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("index json has no weight_map object")?;

    let mut shard_names: Vec<String> = map
        .values()
        .filter_map(|s| s.as_str().map(str::to_string))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut tensors = Vec::new();
    for name in &shard_names {
        let shard_path = dir.join(name);
        let ckpt = load_safetensors(&shard_path)
            .with_context(|| format!("load shard {}", shard_path.display()))?;
        tensors.extend(ckpt.tensors);
    }
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

// ---------------------------------------------------------------------------
// PyTorch pickle
// ---------------------------------------------------------------------------

fn load_pickle(path: &Path, opts: &LoadOptions) -> Result<Checkpoint> {
    let mut named = pickle::read_all_with_key(path, opts.pickle_key.as_deref())
        .with_context(|| format!("read PyTorch checkpoint {}", path.display()))?;

    // Plain `torch.save(model.state_dict())` checkpoints hold tensors at the top
    // level, but wrapper checkpoints (PyTorch Lightning, training states) nest
    // them under "state_dict" — candle finds nothing at the top level for those,
    // so retry with the conventional key before giving up.
    if named.is_empty() && opts.pickle_key.is_none() {
        if let Ok(retried) = pickle::read_all_with_key(path, Some("state_dict")) {
            if !retried.is_empty() {
                eprintln!("note: no top-level tensors — using the \"state_dict\" dict instead");
                named = retried;
            }
        }
    }
    anyhow::ensure!(
        !named.is_empty(),
        "no tensors found in {} (if the checkpoint nests weights under a custom dict, \
         pass --pickle-key <KEY>)",
        path.display()
    );

    let mut tensors = Vec::with_capacity(named.len());
    for (name, tensor) in &named {
        tensors.push(tensor_to_raw(name, tensor)?);
    }
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

// ---------------------------------------------------------------------------
// NumPy
// ---------------------------------------------------------------------------

fn load_npz(path: &Path) -> Result<Checkpoint> {
    let npz = candle_core::npy::NpzTensors::new(path)
        .with_context(|| format!("read npz {}", path.display()))?;
    let mut names = npz.names().into_iter().map(String::from).collect::<Vec<_>>();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    for name in &names {
        let t = npz
            .get(name)?
            .with_context(|| format!("npz entry {name} missing"))?;
        tensors.push(tensor_to_raw(name, &t)?);
    }
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

fn load_npy(path: &Path) -> Result<Checkpoint> {
    let t = Tensor::read_npy(path).with_context(|| format!("read npy {}", path.display()))?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tensor")
        .to_string();
    Ok(Checkpoint { tensors: vec![tensor_to_raw(&name, &t)?], metadata: Vec::new() })
}

// ---------------------------------------------------------------------------
// GGUF (enables re-quantization)
// ---------------------------------------------------------------------------

/// GGUF loading tries the in-crate minimal reader first (covers everything the
/// zsfm writer emits, including BF16, which candle's reader rejects) and falls
/// back to candle's reader for foreign quantization formats (Q4_K etc.).
fn load_gguf(path: &Path) -> Result<Checkpoint> {
    match load_gguf_minimal(path) {
        Ok(ckpt) => Ok(ckpt),
        Err(minimal_err) => load_gguf_candle(path).with_context(|| {
            format!("minimal GGUF reader failed first with: {minimal_err:#}")
        }),
    }
}

fn load_gguf_minimal(path: &Path) -> Result<Checkpoint> {
    use zsfm_gguf::GGUFFile;
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut file = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
    let gguf = GGUFFile::read(&mut file)?;

    let mut tensors = Vec::with_capacity(gguf.tensors.len());
    for info in &gguf.tensors {
        // Stored dims are GGUF order (reversed row-major) — flip back.
        let shape: Vec<u64> = info.shape.iter().rev().copied().collect();
        let (dtype, data) = match info.dtype {
            zsfm_gguf::GGMLType::F32 => (SrcDtype::F32, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::F16 => (SrcDtype::F16, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::BF16 => (SrcDtype::BF16, gguf.tensor_bytes(&mut file, info)?),
            zsfm_gguf::GGMLType::Q8_0 => (
                SrcDtype::F32,
                f32_to_bytes(&gguf.tensor_f32(&mut file, info)?),
            ),
        };
        tensors.push(RawTensor { name: info.name.clone(), shape, dtype, data });
    }
    Ok(Checkpoint { tensors, metadata: gguf.metadata })
}

fn load_gguf_candle(path: &Path) -> Result<Checkpoint> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut file = BufReader::with_capacity(zsfm_gguf::READ_BUF_CAPACITY, file);
    let content = gguf_file::Content::read(&mut file).context("read GGUF header")?;
    let device = Device::Cpu;

    let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    names.sort();

    let mut tensors = Vec::with_capacity(names.len());
    for name in &names {
        let qt = content
            .tensor(&mut file, name, &device)
            .with_context(|| format!("read tensor {name}"))?;
        let t = qt.dequantize(&device)?;
        tensors.push(tensor_to_raw(name, &t)?);
    }

    let mut keys: Vec<&String> = content.metadata.keys().collect();
    keys.sort();
    let mut metadata = Vec::new();
    for key in keys {
        match gguf_value_to_meta(&content.metadata[key]) {
            Some(v) => metadata.push((key.clone(), v)),
            None => eprintln!("note: skipping GGUF metadata key {key} (unsupported value type)"),
        }
    }

    Ok(Checkpoint { tensors, metadata })
}

fn gguf_value_to_meta(v: &gguf_file::Value) -> Option<GGUFMetaValue> {
    use gguf_file::Value as V;
    Some(match v {
        V::U8(x) => GGUFMetaValue::Uint8(*x),
        V::I8(x) => GGUFMetaValue::Int8(*x),
        V::U16(x) => GGUFMetaValue::Uint16(*x),
        V::I16(x) => GGUFMetaValue::Int16(*x),
        V::U32(x) => GGUFMetaValue::Uint32(*x),
        V::I32(x) => GGUFMetaValue::Int32(*x),
        V::U64(x) => GGUFMetaValue::Uint64(*x),
        V::I64(x) => GGUFMetaValue::Int64(*x),
        V::F32(x) => GGUFMetaValue::Float32(*x),
        V::F64(x) => GGUFMetaValue::Float64(*x),
        V::Bool(x) => GGUFMetaValue::Bool(*x),
        V::String(x) => GGUFMetaValue::String(x.clone()),
        V::Array(items) => {
            if items.iter().all(|i| matches!(i, V::U32(_))) {
                GGUFMetaValue::ArrayUint32(
                    items.iter().filter_map(|i| i.to_u32().ok()).collect(),
                )
            } else if items.iter().all(|i| matches!(i, V::F32(_))) {
                GGUFMetaValue::ArrayFloat32(
                    items.iter().filter_map(|i| i.to_f32().ok()).collect(),
                )
            } else if items.iter().all(|i| matches!(i, V::String(_))) {
                GGUFMetaValue::ArrayString(
                    items
                        .iter()
                        .filter_map(|i| i.to_string().ok().cloned())
                        .collect(),
                )
            } else {
                return None;
            }
        }
    })
}

// ---------------------------------------------------------------------------
// candle Tensor → RawTensor
// ---------------------------------------------------------------------------

fn tensor_to_raw(name: &str, t: &Tensor) -> Result<RawTensor> {
    let shape: Vec<u64> = t.dims().iter().map(|&d| d as u64).collect();
    let flat = t.flatten_all()?;
    let (dtype, data) = match t.dtype() {
        DType::F32 => (SrcDtype::F32, f32_to_bytes(&flat.to_vec1::<f32>()?)),
        DType::F16 => {
            let vals = flat.to_vec1::<half::f16>()?;
            (
                SrcDtype::F16,
                vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
            )
        }
        DType::BF16 => {
            let vals = flat.to_vec1::<half::bf16>()?;
            (
                SrcDtype::BF16,
                vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
            )
        }
        other => {
            eprintln!("note: tensor {name}: casting {other:?} to F32");
            let vals = flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
    };
    Ok(RawTensor { name: name.to_string(), shape, dtype, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_safetensors(path: &Path) {
        let dev = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &dev).unwrap();
        let b = Tensor::from_vec((0..32).map(|i| i as f32).collect::<Vec<_>>(), 32, &dev).unwrap();
        candle_core::safetensors::save(
            &std::collections::HashMap::from([("a".to_string(), a), ("b".to_string(), b)]),
            path,
        )
        .unwrap();
    }

    #[test]
    fn safetensors_roundtrip_and_shape_order() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        write_test_safetensors(&path);

        let ckpt = load_checkpoint(&[path], &LoadOptions::default()).unwrap();
        assert_eq!(ckpt.tensors.len(), 2);
        let a = ckpt.tensors.iter().find(|t| t.name == "a").unwrap();
        assert_eq!(a.shape, vec![2, 3]); // python order
        assert_eq!(a.dtype, SrcDtype::F32);
        let vals: Vec<f32> = a
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn duplicate_names_rejected() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dup.safetensors");
        write_test_safetensors(&path);
        let err = match load_checkpoint(&[path.clone(), path], &LoadOptions::default()) {
            Ok(_) => panic!("expected duplicate-name error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("duplicate tensor name"));
    }

    #[test]
    fn sniffs_safetensors_without_extension() {
        let dir = std::env::temp_dir().join("zsfm-checkpoint-test");
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("noext-src.safetensors");
        write_test_safetensors(&st);
        let noext = dir.join("noext_checkpoint");
        std::fs::copy(&st, &noext).unwrap();
        let ckpt = load_checkpoint(&[noext], &LoadOptions::default()).unwrap();
        assert_eq!(ckpt.tensors.len(), 2);
    }
}
