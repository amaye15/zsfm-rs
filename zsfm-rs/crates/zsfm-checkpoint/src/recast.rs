//! Read any supported checkpoint (including an existing GGUF) and rewrite it at a
//! different dtype, carrying metadata through. Shared by the generic `zsfm convert`
//! command and by every per-model CLI's cached-GGUF fast path: once a repo has been
//! downloaded and converted to a canonical F32 GGUF once, later requests for a
//! different dtype recast from that cache instead of re-downloading from HuggingFace.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};

use zsfm_gguf::{GGMLType, GGUFWriter};

use crate::cast;
use crate::read::{load_checkpoint, LoadOptions};

pub fn recast(input: &Path, output: &Path, dtype: GGMLType) -> Result<()> {
    let ckpt = load_checkpoint(&[input.to_path_buf()], &LoadOptions::default())
        .with_context(|| format!("load checkpoint {}", input.display()))?;
    anyhow::ensure!(!ckpt.tensors.is_empty(), "checkpoint {} contains no tensors", input.display());

    let mut writer = GGUFWriter::new();
    for (k, v) in ckpt.metadata {
        writer.add_metadata(k, v);
    }

    let mut fallback_count = 0usize;
    for t in &ckpt.tensors {
        let n_elems: u64 = t.shape.iter().product();
        let innermost = t.shape.last().copied().unwrap_or(1);
        // Tensors too small for Q8_0's 32-element blocks fall back to F32, matching
        // the generic `zsfm convert` command's behavior.
        let dst = if dtype == GGMLType::Q8_0 && (innermost % 32 != 0 || n_elems % 32 != 0) {
            fallback_count += 1;
            GGMLType::F32
        } else {
            dtype
        };
        let data =
            cast::cast_data(&t.data, t.dtype, dst).with_context(|| format!("tensor {}: cast failed", t.name))?;
        let gguf_shape: Vec<u64> = t.shape.iter().rev().copied().collect();
        writer.add_tensor(t.name.clone(), gguf_shape, dst, data);
    }
    if fallback_count > 0 {
        eprintln!("note: {fallback_count} tensor(s) fell back to F32 (too small for Q8_0 blocks)");
    }

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let out_file = File::create(output).with_context(|| format!("create {}", output.display()))?;
    writer.write_to(&mut BufWriter::new(out_file))?;
    Ok(())
}
