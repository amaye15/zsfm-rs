use serde::Deserialize;

/// Top-level FlowState `config.json`.
#[derive(Debug, Deserialize, Clone)]
pub struct FlowStateConfig {
    pub context_length: u32,
    pub decoder_dim: u32,
    pub decoder_patch_len: u32,
    pub decoder_type: String,
    pub embedding_feature_dim: u32,
    pub encoder_num_hippo_blocks: u32,
    pub encoder_num_layers: u32,
    pub encoder_state_dim: u32,
    pub quantiles: Vec<f32>,
    #[serde(default = "default_bool_true")]
    pub with_missing: bool,
    #[serde(default = "default_bool_true")]
    pub use_freq: bool,
    #[serde(default = "default_bool_true")]
    pub init_processing: bool,
    #[serde(default = "default_u32_2048")]
    pub min_context: u32,
}

impl FlowStateConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn n_quantiles(&self) -> u32 {
        self.quantiles.len() as u32
    }

    /// Number of input channels (value + missing mask).
    pub fn n_inputs(&self) -> u32 {
        if self.with_missing { 2 } else { 1 }
    }

    /// Legendre basis range for "legs" / "hlegs" decoder.
    pub fn basis_range(&self) -> [f32; 2] {
        let dt = self.decoder_type.to_lowercase();
        if dt == "hlegs" {
            [0.0, 0.95]
        } else {
            [-1.0, 0.95]
        }
    }
}

fn default_bool_true() -> bool { true }
fn default_u32_2048() -> u32 { 2048 }
