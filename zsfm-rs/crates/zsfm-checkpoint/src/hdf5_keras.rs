//! HDF5 (`.h5` / `.hdf5`) and Keras v3 (`.keras`) checkpoint loading.
//!
//! HDF5 files are walked recursively: every numeric dataset becomes a tensor
//! named by its group path with `/` separators replaced by `.` (e.g.
//! `/model_weights/dense/kernel:0` → `model_weights.dense.kernel:0`).
//! F32 and F16 datasets are preserved exactly; F64, integer, and bool datasets
//! are cast to F32 with a note; non-numeric datasets (strings, compounds) are
//! skipped with a note.
//!
//! `.keras` archives are zip files wrapping `model.weights.h5` — the embedded
//! weights file is extracted to a temp location and parsed the same way.

use std::path::Path;

use anyhow::{Context, Result};
use hdf5::types::{FloatSize, TypeDescriptor};

use crate::cast::{f32_to_bytes, SrcDtype};
use crate::read::{Checkpoint, RawTensor};

pub fn load_hdf5(path: &Path) -> Result<Checkpoint> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("open HDF5 file {}", path.display()))?;
    let mut tensors = Vec::new();
    walk_group(&file, &mut tensors)?;
    anyhow::ensure!(
        !tensors.is_empty(),
        "no numeric datasets found in {}",
        path.display()
    );
    Ok(Checkpoint { tensors, metadata: Vec::new() })
}

fn walk_group(group: &hdf5::Group, out: &mut Vec<RawTensor>) -> Result<()> {
    let mut datasets = group.datasets().unwrap_or_default();
    datasets.sort_by_key(|d| d.name());
    for ds in datasets {
        match dataset_to_raw(&ds) {
            Ok(Some(t)) => out.push(t),
            Ok(None) => {}
            Err(e) => eprintln!("note: skipping dataset {}: {e:#}", ds.name()),
        }
    }
    let mut subgroups = group.groups().unwrap_or_default();
    subgroups.sort_by_key(|g| g.name());
    for sub in subgroups {
        walk_group(&sub, out)?;
    }
    Ok(())
}

fn dataset_to_raw(ds: &hdf5::Dataset) -> Result<Option<RawTensor>> {
    let name = ds.name().trim_start_matches('/').replace('/', ".");
    let shape: Vec<u64> = ds.shape().iter().map(|&d| d as u64).collect();
    let descriptor = ds.dtype()?.to_descriptor()?;

    let (dtype, data): (SrcDtype, Vec<u8>) = match descriptor {
        TypeDescriptor::Float(FloatSize::U4) => {
            (SrcDtype::F32, f32_to_bytes(&ds.read_raw::<f32>()?))
        }
        #[allow(unreachable_patterns)] // U2 only exists with the f16 feature
        TypeDescriptor::Float(FloatSize::U2) => {
            let vals = ds.read_raw::<half::f16>()?;
            (
                SrcDtype::F16,
                vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
            )
        }
        TypeDescriptor::Float(FloatSize::U8) => {
            eprintln!("note: dataset {name}: casting F64 to F32");
            let vals: Vec<f32> = ds.read_raw::<f64>()?.iter().map(|&v| v as f32).collect();
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        TypeDescriptor::Integer(size) => {
            eprintln!("note: dataset {name}: casting signed integer to F32");
            let vals: Vec<f32> = match size {
                hdf5::types::IntSize::U1 => {
                    ds.read_raw::<i8>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U2 => {
                    ds.read_raw::<i16>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U4 => {
                    ds.read_raw::<i32>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U8 => {
                    ds.read_raw::<i64>()?.iter().map(|&v| v as f32).collect()
                }
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        TypeDescriptor::Unsigned(size) => {
            eprintln!("note: dataset {name}: casting unsigned integer to F32");
            let vals: Vec<f32> = match size {
                hdf5::types::IntSize::U1 => {
                    ds.read_raw::<u8>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U2 => {
                    ds.read_raw::<u16>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U4 => {
                    ds.read_raw::<u32>()?.iter().map(|&v| v as f32).collect()
                }
                hdf5::types::IntSize::U8 => {
                    ds.read_raw::<u64>()?.iter().map(|&v| v as f32).collect()
                }
            };
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        TypeDescriptor::Boolean => {
            eprintln!("note: dataset {name}: casting bool to F32");
            let vals: Vec<f32> = ds
                .read_raw::<bool>()?
                .iter()
                .map(|&v| v as u8 as f32)
                .collect();
            (SrcDtype::F32, f32_to_bytes(&vals))
        }
        other => {
            eprintln!("note: skipping non-numeric dataset {name} ({other})");
            return Ok(None);
        }
    };

    Ok(Some(RawTensor { name, shape, dtype, data }))
}

// ---------------------------------------------------------------------------
// Keras v3 (.keras = zip wrapping model.weights.h5)
// ---------------------------------------------------------------------------

pub fn load_keras(path: &Path) -> Result<Checkpoint> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file).context("read .keras zip archive")?;

    let weights_entry = archive
        .file_names()
        .find(|n| n.ends_with(".weights.h5"))
        .or_else(|| archive.file_names().find(|n| n.ends_with(".h5")))
        .map(String::from)
        .context(".keras archive contains no .weights.h5 entry")?;

    // libhdf5 needs a real file path, so extract the weights to a temp file.
    let tmp = std::env::temp_dir().join(format!(
        "zsfm-keras-{}-{}.h5",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    {
        let mut entry = archive.by_name(&weights_entry)?;
        let mut out = std::fs::File::create(&tmp)
            .with_context(|| format!("create temp file {}", tmp.display()))?;
        std::io::copy(&mut entry, &mut out).context("extract weights from .keras archive")?;
    }
    let result = load_hdf5(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result.with_context(|| format!("parse {weights_entry} from .keras archive"))
}
