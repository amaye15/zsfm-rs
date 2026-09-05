//! Helpers shared by more than one model's CLI module.

use std::path::Path;

use anyhow::Result;
use zsfm_checkpoint::{load_checkpoint, LoadOptions};

/// Shared `inspect-tensors` implementation: print every tensor's name, shape,
/// and dtype from a local checkpoint file. Format is auto-detected — works
/// whether the model's raw download is safetensors (most models), a raw
/// PyTorch pickle `.ckpt` (lag_llama, tirex), or a GGUF file, via the same
/// format-agnostic loader `zsfm convert`/`zsfm inspect` use.
pub fn inspect_tensors(path: &Path) -> Result<()> {
    let ckpt = load_checkpoint(&[path.to_path_buf()], &LoadOptions::default())?;
    println!("Tensors in {}:", path.display());
    let mut tensors: Vec<_> = ckpt.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for t in tensors {
        println!("  {:80} {} {:?}", t.name, t.dtype.name(), t.shape);
    }
    Ok(())
}

/// Delete a model's cached files (canonical GGUF + config) and optionally an
/// output GGUF. `canonical` should be the path returned by
/// `zsfm_hub::canonical_gguf_path` (or the variant-aware equivalent for
/// mitra/tabfm). `output` is an optional additional GGUF to remove.
pub fn delete_cached_model(canonical: &Path, output: Option<&Path>) -> Result<()> {
    let mut deleted_any = false;

    // The canonical file lives at `<model_dir>/<owner>__<name>/model-f32.gguf`;
    // its parent is the per-model cache directory (contains config.json, etc.).
    if let Some(cache_dir) = canonical.parent() {
        if cache_dir.exists() {
            // Count files for a nicer message before deleting.
            let file_count = walk_file_count(cache_dir);
            std::fs::remove_dir_all(cache_dir)?;
            println!("Deleted cache directory {} ({} files)", cache_dir.display(), file_count);
            deleted_any = true;
        } else if canonical.exists() {
            std::fs::remove_file(canonical)?;
            println!("Deleted {}", canonical.display());
            deleted_any = true;
        } else {
            println!("No cache found at {} (already deleted?)", cache_dir.display());
        }
    } else if canonical.exists() {
        std::fs::remove_file(canonical)?;
        println!("Deleted {}", canonical.display());
        deleted_any = true;
    }

    if let Some(out) = output {
        if out.exists() {
            std::fs::remove_file(out)?;
            println!("Deleted output {}", out.display());
            deleted_any = true;
        } else {
            println!("Output file not found: {} (already deleted?)", out.display());
        }
    }

    if !deleted_any {
        println!("Nothing to delete.");
    }
    Ok(())
}

fn walk_file_count(dir: &Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += walk_file_count(&path);
            } else {
                count += 1;
            }
        }
    }
    count
}
