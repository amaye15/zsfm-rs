use std::path::Path;

use crate::log::log_status;

/// Delete a model's cached files (canonical GGUF + config) and optionally an
/// output GGUF. Shared by the `zsfm` CLI and the Python bindings so both
/// report identical messages on stdout/stderr.
pub fn delete_cached_model(canonical: &Path, output: Option<&Path>) -> anyhow::Result<()> {
    let mut deleted_any = false;

    if let Some(cache_dir) = canonical.parent() {
        if cache_dir.exists() {
            let file_count = walk_file_count(cache_dir);
            std::fs::remove_dir_all(cache_dir)?;
            log_status(&format!(
                "Deleted cache directory {} ({} files)",
                cache_dir.display(),
                file_count
            ));
            deleted_any = true;
        } else if canonical.exists() {
            std::fs::remove_file(canonical)?;
            log_status(&format!("Deleted {}", canonical.display()));
            deleted_any = true;
        } else {
            log_status(&format!(
                "No cache found at {} (already deleted?)",
                cache_dir.display()
            ));
        }
    } else if canonical.exists() {
        std::fs::remove_file(canonical)?;
        log_status(&format!("Deleted {}", canonical.display()));
        deleted_any = true;
    }

    if let Some(out) = output {
        if out.exists() {
            std::fs::remove_file(out)?;
            log_status(&format!("Deleted output {}", out.display()));
            deleted_any = true;
        } else {
            log_status(&format!(
                "Output file not found: {} (already deleted?)",
                out.display()
            ));
        }
    }

    if !deleted_any {
        log_status("Nothing to delete.");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn deletes_cache_dir_and_output() {
        let base = std::env::temp_dir().join(format!("zsfm-del-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let cache_dir = base.join("owner__repo");
        fs::create_dir_all(&cache_dir).unwrap();
        let canonical = cache_dir.join("model-f32.gguf");
        fs::write(&canonical, b"gguf").unwrap();
        let out = base.join("out.gguf");
        fs::write(&out, b"out").unwrap();
        delete_cached_model(&canonical, Some(&out)).unwrap();
        assert!(!cache_dir.exists());
        assert!(!out.exists());
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn missing_cache_is_ok() {
        let base = std::env::temp_dir().join(format!("zsfm-del-miss-{}", std::process::id()));
        let canonical = base.join("owner__repo").join("model-f32.gguf");
        delete_cached_model(&canonical, None).unwrap();
    }
}
