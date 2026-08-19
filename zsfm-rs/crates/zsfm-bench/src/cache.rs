use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::eval::Metrics;

#[derive(Clone, Serialize, Deserialize)]
pub struct SoloResult {
    pub metrics: Metrics,
    #[serde(with = "crate::eval::nan_null")]
    pub lat_ms: f64,
    pub errors: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub name: String,
    pub metrics: Metrics,
}

/// Structured result for one (dataset, context, horizon, windows) config —
/// replaces the old `.bench_cache.json`'s raw captured stdout text (which
/// `gen_report.py` had to regex-parse back apart) with the actual typed
/// results, computed once, directly reusable by the report writer.
#[derive(Clone, Serialize, Deserialize)]
pub struct CachedRun {
    pub series_len: usize,
    pub test_rows: usize,
    pub solo: HashMap<String, SoloResult>,
    pub ensembles: Vec<EnsembleResult>,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Cache {
    #[serde(flatten)]
    entries: HashMap<String, CachedRun>,
}

impl Cache {
    pub fn load(path: &Path) -> Cache {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Cache::default();
        };
        match serde_json::from_str(&text) {
            Ok(cache) => cache,
            Err(e) => {
                eprintln!(
                    "warning: {} exists but failed to parse ({e}) — starting with an empty cache",
                    path.display()
                );
                Cache::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))
    }

    pub fn get(&self, key: &str) -> Option<&CachedRun> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: String, run: CachedRun) {
        self.entries.insert(key, run);
    }
}

pub fn key(dataset: &str, context: usize, horizon: usize, windows: usize) -> String {
    format!("{dataset}:ctx{context}:h{horizon}:w{windows}")
}
