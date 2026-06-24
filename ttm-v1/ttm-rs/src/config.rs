use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TtmConfig {
    pub context_length: usize,
    pub prediction_length: usize,
    pub patch_length: usize,
    pub patch_stride: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub decoder_d_model: usize,
    pub decoder_num_layers: usize,
    pub expansion_factor: usize,
    pub adaptive_patching_levels: usize,
    pub num_patches: usize,
    /// "std", "mean", or "none"
    #[serde(default = "default_scaling")]
    pub scaling: String,
    #[serde(default)]
    pub gated_attn: bool,
    #[serde(default)]
    pub norm_eps: f64,
}

fn default_scaling() -> String { "std".into() }

impl TtmConfig {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("parse TtmConfig")
    }
}
