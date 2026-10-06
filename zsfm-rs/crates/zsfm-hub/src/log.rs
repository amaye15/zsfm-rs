/// Status logging that respects `ZSFM_QUIET`.
///
/// All human-readable progress goes to stderr so `infer` JSON on stdout stays
/// pipe-clean. Set `ZSFM_QUIET=1` (or pass `zsfm --quiet`) to silence status
/// lines; errors still print.
pub fn log_status(msg: &str) {
    if std::env::var_os("ZSFM_QUIET").is_some() {
        return;
    }
    eprintln!("{msg}");
}

/// Typed Hub errors. Functions still return `anyhow::Result` at the boundary
/// so callers keep one error type, but raise these variants so messages stay
/// consistent and matchable via `.to_string()`.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error(
        "resume offset mismatch for {relpath}: local has {local} bytes, server starts at {server}"
    )]
    ResumeMismatch {
        relpath: String,
        local: u64,
        server: u64,
    },
    #[error("size mismatch for {relpath}: server said {expected} bytes, got {actual}")]
    SizeMismatch {
        relpath: String,
        expected: u64,
        actual: u64,
    },
    #[error("shard filename collision after flattening: {0}")]
    ShardCollision(String),
    #[error("repo listing too large: {0} bytes")]
    ListingTooLarge(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages() {
        let e = HubError::SizeMismatch {
            relpath: "model.safetensors".into(),
            expected: 10,
            actual: 9,
        };
        assert!(e.to_string().contains("size mismatch"));
    }
}
