use serde::Deserialize;

/// Mirrors the relevant fields from Datadog/Toto-2.0 config.json.
/// Accepts the Python model field names (d_model, num_layers, …).
/// Unknown fields are ignored so we stay forward-compatible.
#[derive(Debug, Deserialize)]
pub struct TotoConfig {
    /// Number of transformer layers.
    #[serde(alias = "num_layers", default = "default_u32::<48>")]
    pub num_hidden_layers: u32,

    /// Hidden / embedding dimension.
    #[serde(alias = "d_model", default = "default_u32::<2048>")]
    pub hidden_size: u32,

    /// Number of query attention heads.
    #[serde(alias = "num_heads", default = "default_u32::<32>")]
    pub num_attention_heads: u32,

    /// Number of KV groups (num_groups in Python = GQA groups).
    #[serde(alias = "num_groups", default = "default_u32::<32>")]
    pub num_key_value_heads: u32,

    /// Per-head QK dimension.
    #[serde(alias = "qk_dim", default = "default_u32::<64>")]
    pub head_dim: u32,

    /// Patch size in timesteps.
    #[serde(default = "default_u32::<32>")]
    pub patch_size: u32,

    /// Number of output quantile levels.
    #[serde(default = "default_u32::<9>")]
    pub num_quantiles: u32,
}

impl TotoConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

fn default_u32<const N: u32>() -> u32 { N }
