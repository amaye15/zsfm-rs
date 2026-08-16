use serde::Deserialize;

/// `{task}/config.json` from `google/tabfm-1.0.0-pytorch`. Mirrors the upstream Python loader's
/// `Config` dataclass (`tabfm/src/pytorch/tabfm_v1_0_0.py`) field-for-field.
#[derive(Debug, Deserialize)]
pub struct TabFMConfig {
    pub embed_dim: u32,
    pub max_classes: u32,
    pub col_num_blocks: u32,
    pub col_nhead: u32,
    pub col_num_inds: u32,
    pub row_num_blocks: u32,
    pub row_nhead: u32,
    pub row_num_cls: u32,
    pub icl_num_blocks: u32,
    pub icl_nhead: u32,
    pub ff_factor: u32,
    pub feature_group_size: u32,
    pub is_classifier: bool,
    #[serde(default = "default_num_freq")]
    pub num_freq: u32,
    /// `null` in the JSON means "use the `TabFM.__init__` default of `icl_dim * 2`".
    #[serde(default)]
    pub decoder_hidden: Option<u32>,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

impl TabFMConfig {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    /// `d_model` fed into the ICL stage: row_num_cls CLS-token embeddings, concatenated.
    pub fn icl_dim(&self) -> u32 {
        self.embed_dim * self.row_num_cls
    }

    pub fn col_dim_ff(&self) -> u32 {
        self.embed_dim * self.ff_factor
    }

    pub fn icl_dim_ff(&self) -> u32 {
        self.icl_dim() * self.ff_factor
    }

    pub fn decoder_hidden(&self) -> u32 {
        self.decoder_hidden.unwrap_or(self.icl_dim() * 2)
    }
}

fn default_num_freq() -> u32 { 32 }
fn default_norm_eps() -> f64 { 1e-6 }
