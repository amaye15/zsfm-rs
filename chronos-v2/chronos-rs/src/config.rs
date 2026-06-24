use serde::Deserialize;

/// Top-level Chronos-2 `config.json`.
/// The `chronos_config` key holds a nested dict with forecasting-specific settings.
#[derive(Debug, Deserialize)]
pub struct Chronos2Config {
    pub d_model: u32,
    #[serde(default = "default_u32::<64>")]
    pub d_kv: u32,
    pub d_ff: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_f64_ten_thousand")]
    pub rope_theta: f64,
    #[serde(default = "default_str_relu")]
    pub feed_forward_proj: String,
    pub chronos_config: ChronosInnerConfig,
}

/// Nested `chronos_config` dict inside `config.json`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChronosInnerConfig {
    pub context_length: u32,
    pub input_patch_size: u32,
    pub output_patch_size: u32,
    pub input_patch_stride: u32,
    pub quantiles: Vec<f32>,
    #[serde(default)]
    pub use_reg_token: bool,
    #[serde(default)]
    pub use_arcsinh: bool,
    #[serde(default = "default_u32::<1>")]
    pub max_output_patches: u32,
    pub time_encoding_scale: Option<u32>,
}

impl Chronos2Config {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// Effective time_encoding_scale: falls back to context_length if not set.
    pub fn time_encoding_scale(&self) -> u32 {
        self.chronos_config
            .time_encoding_scale
            .unwrap_or(self.chronos_config.context_length)
    }

    /// Dense activation function name (e.g. "relu").
    pub fn dense_act_fn(&self) -> &str {
        // feed_forward_proj may be "relu", "gelu", or "gated-gelu" etc.
        // Chronos-2 asserts not gated, so just split on '-' and take the last part.
        self.feed_forward_proj.split('-').last().unwrap_or("relu")
    }
}

fn default_u32<const N: u32>() -> u32 { N }
fn default_f64_ten_thousand() -> f64 { 10000.0 }
fn default_layer_norm_eps() -> f64 { 1e-6 }
fn default_str_relu() -> String { "relu".into() }
