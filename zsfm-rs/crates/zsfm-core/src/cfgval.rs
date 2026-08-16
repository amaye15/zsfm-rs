//! Small helpers for pulling `InferConfig` fields out of a HuggingFace `config.json`
//! `serde_json::Value` with a default fallback — the same `.as_u64().unwrap_or(N) as usize`
//! shape every model's config-loading code (CLI, and later Python bindings) needs.

pub fn json_usize(v: &serde_json::Value, key: &str, default: usize) -> usize {
    v[key].as_u64().map(|n| n as usize).unwrap_or(default)
}

pub fn json_f64(v: &serde_json::Value, key: &str, default: f64) -> f64 {
    v[key].as_f64().unwrap_or(default)
}

pub fn json_bool(v: &serde_json::Value, key: &str, default: bool) -> bool {
    v[key].as_bool().unwrap_or(default)
}
