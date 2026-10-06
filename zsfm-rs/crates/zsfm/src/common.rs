//! Helpers shared by more than one model's CLI module.

use std::path::Path;

use anyhow::Result;
use clap::ValueEnum;
use zsfm_checkpoint::{load_checkpoint, LoadOptions};
use zsfm_gguf::GGMLType;

/// Shared `--dtype` flag. BF16 is deliberately absent: candle 0.8's GGUF reader
/// (used by every model's own `infer` loader) cannot parse ggml dtype 30, so a
/// bf16 file would fail to load back through the same model's `infer` command.
/// The format-agnostic `zsfm convert` path supports BF16 for other consumers.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DtypeArg {
    F32,
    F16,
    Q8,
}

impl DtypeArg {
    pub fn name(self) -> &'static str {
        match self {
            DtypeArg::F32 => "f32",
            DtypeArg::F16 => "f16",
            DtypeArg::Q8 => "q8",
        }
    }
}

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => GGMLType::F32,
            DtypeArg::F16 => GGMLType::F16,
            DtypeArg::Q8 => GGMLType::Q8_0,
        }
    }
}

/// Return the canonical cache path, recasting from cache when present.
///
/// Returns `true` when the caller can return early (cache hit already
/// recast to `output`), `false` when the caller must download + convert.
/// Thin wrapper over [`zsfm_hub::try_recast_from_cache`] so CLI + Python share
/// one implementation.
pub fn try_recast_from_cache(
    canonical: &Path,
    output: &Path,
    dtype: GGMLType,
    redownload: bool,
) -> Result<bool> {
    zsfm_hub::try_recast_from_cache(canonical, output, dtype, redownload)
}

/// Index of the median (q0.5) quantile level, with center fallback.
pub fn median_index(quantile_levels: &[f32]) -> usize {
    quantile_levels
        .iter()
        .position(|&q| (q - 0.5).abs() < 1e-6)
        .unwrap_or(quantile_levels.len() / 2)
}

/// Shared `inspect-tensors` implementation: print every tensor's name, shape,
/// and dtype from a local checkpoint file. Format is auto-detected — works
/// whether the model's raw download is safetensors (most models), a raw
/// PyTorch pickle `.ckpt` (lag_llama, tirex), or a GGUF file, via the same
/// format-agnostic loader `zsfm convert`/`zsfm inspect` use.
pub fn inspect_tensors(path: &Path) -> Result<()> {
    let ckpt = load_checkpoint(&[path.to_path_buf()], &LoadOptions::default())?;
    eprintln!("Tensors in {}:", path.display());
    let mut tensors: Vec<_> = ckpt.tensors.iter().collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    for t in tensors {
        println!("  {:80} {} {:?}", t.name, t.dtype.name(), t.shape);
    }
    Ok(())
}

/// Delete a model's cached files (canonical GGUF + config) and optionally an
/// output GGUF. Thin wrapper over [`zsfm_hub::delete_cached_model`] so all
/// CLI modules share one import path (`crate::common`).
pub fn delete_cached_model(canonical: &Path, output: Option<&Path>) -> Result<()> {
    zsfm_hub::delete_cached_model(canonical, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_names() {
        assert_eq!(DtypeArg::F32.name(), "f32");
        assert_eq!(DtypeArg::F16.name(), "f16");
        assert_eq!(DtypeArg::Q8.name(), "q8");
        assert_eq!(GGMLType::from(DtypeArg::F32), GGMLType::F32);
        assert_eq!(GGMLType::from(DtypeArg::Q8), GGMLType::Q8_0);
    }

    #[test]
    fn median_prefers_q05() {
        let levels = vec![0.1, 0.5, 0.9];
        assert_eq!(median_index(&levels), 1);
        let no_median = vec![0.1, 0.9];
        assert_eq!(median_index(&no_median), 1);
        let empty: Vec<f32> = vec![];
        assert_eq!(median_index(&empty), 0);
    }

    #[test]
    fn cache_miss_returns_false() {
        let dir = std::env::temp_dir().join(format!("zsfm-common-{}", std::process::id()));
        let canonical = dir.join("owner__repo").join("model-f32.gguf");
        let out = dir.join("out.gguf");
        // No file exists → miss, no I/O.
        assert!(!try_recast_from_cache(&canonical, &out, GGMLType::F16, false).unwrap());
        // Redownload forces miss even if file existed (still missing here).
        assert!(!try_recast_from_cache(&canonical, &out, GGMLType::F16, true).unwrap());
    }
}
