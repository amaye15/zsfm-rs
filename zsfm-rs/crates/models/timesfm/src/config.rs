/// Hardcoded architecture constants for TimesFM 2.5 200M.
///
/// These match `TimesFM_2p5_200M_Definition` in the Python source exactly.
/// No config.json is needed — the architecture is fixed for this model.
#[derive(Debug, Clone)]
pub struct TimesFMConfig {
    /// Input patch length (p = 32).
    pub input_patch_len: usize,
    /// Output patch length (o = 128).
    pub output_patch_len: usize,
    /// Output quantile length for the continuous quantile head (os = 1024).
    pub output_quantile_len: usize,
    /// Number of stacked transformer layers.
    pub num_layers: usize,
    /// Transformer model dimension.
    pub d_model: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Head dimension = d_model / num_heads.
    pub head_dim: usize,
    /// Quantile levels predicted (excludes the implicit point forecast at index 0).
    pub quantiles: Vec<f32>,
    /// Total number of per-timestep outputs = len(quantiles) + 1 (point).
    pub n_outputs: usize,
    /// Index used to extract point/AR forecast from the 10-dim output.
    pub decode_index: usize,
    /// Maximum context tokens the model accepts (in individual timesteps).
    pub context_limit: usize,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency (theta = 10000).
    pub rope_theta: f64,
}

impl Default for TimesFMConfig {
    fn default() -> Self {
        let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let n_outputs = quantiles.len() + 1; // 10
        Self {
            input_patch_len:    32,
            output_patch_len:   128,
            output_quantile_len: 1024,
            num_layers:         20,
            d_model:            1280,
            d_ff:               1280,
            num_heads:          16,
            head_dim:           80,
            n_outputs,
            decode_index:       5,
            context_limit:      16384,
            rms_norm_eps:       1e-6,
            rope_theta:         10000.0,
            quantiles,
        }
    }
}

impl TimesFMConfig {
    pub fn new() -> Self {
        Self::default()
    }
}
