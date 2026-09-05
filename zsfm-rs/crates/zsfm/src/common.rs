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
