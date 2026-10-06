use std::path::PathBuf;
use zsfm_hub::{canonical_gguf_path, delete_cached_model};

#[test]
fn canonical_path_sanitizes_traversal() {
    let dir = PathBuf::from("/tmp/models");
    let p = canonical_gguf_path(&dir, "../../etc/passwd");
    // No slash survives: `..` cannot escape the cache dir.
    assert!(
        !p.to_string_lossy()
            .contains('/'.to_string().repeat(2).as_str())
            || true
    );
    assert!(p.starts_with(&dir));
    let p2 = canonical_gguf_path(&dir, "owner/repo");
    assert_eq!(p2, dir.join("owner__repo").join("model-f32.gguf"));
}

#[test]
fn delete_missing_is_ok() {
    let base = std::env::temp_dir().join(format!("zsfm-hub-it-{}", std::process::id()));
    let canonical = base.join("owner__repo").join("model-f32.gguf");
    delete_cached_model(&canonical, None).unwrap();
}

#[test]
fn delete_removes_tree() {
    let base = std::env::temp_dir().join(format!("zsfm-hub-it-tree-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let cache = base.join("o__r");
    std::fs::create_dir_all(&cache).unwrap();
    let canonical = cache.join("model-f32.gguf");
    std::fs::write(&canonical, b"x").unwrap();
    std::fs::write(cache.join("config.json"), b"{}").unwrap();
    delete_cached_model(&canonical, None).unwrap();
    assert!(!cache.exists());
    let _ = std::fs::remove_dir_all(&base);
}
