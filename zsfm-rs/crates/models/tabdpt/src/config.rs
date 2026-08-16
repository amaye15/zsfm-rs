/// TabDPT model configuration. One checkpoint (`Layer6/TabDPT`) serves both classification and
/// regression — the head produces `max_num_classes + regression_bin_count` outputs; callers
/// slice whichever half they need.
#[derive(Clone, Debug)]
pub struct TabDptConfig {
    pub dim: usize,                  // 512 (emsize / ninp)
    pub n_layers: usize,             // 32
    pub n_heads: usize,              // 8
    pub ff_dim: usize,               // 512 (nhid)
    pub y_encoder_dim: usize,        // 128
    pub max_num_classes: usize,      // 16 (n_out)
    pub regression_bin_count: usize, // 2048
    pub regression_bin_min: f32,     // -10.0
    pub regression_bin_max: f32,     // 10.0
    pub max_num_features: usize,     // 128
    pub base_len: usize,             // 64 (min_eval_context)
    pub max_len: usize,              // 1_048_576 (max_eval_context)
    pub n_thinking_rows: usize,      // 64
}

impl TabDptConfig {
    /// Matches `Layer6/TabDPT`'s `tabdpt1_2.safetensors` embedded config exactly.
    pub fn default_v1_2() -> Self {
        Self {
            dim: 512,
            n_layers: 32,
            n_heads: 8,
            ff_dim: 512,
            y_encoder_dim: 128,
            max_num_classes: 16,
            regression_bin_count: 2048,
            regression_bin_min: -10.0,
            regression_bin_max: 10.0,
            max_num_features: 128,
            base_len: 64,
            max_len: 1_048_576,
            n_thinking_rows: 64,
        }
    }

    /// `kappa = (sqrt(head_dim) - 1) / ln(max_len / base_len)`, used by every layer's attention
    /// temperature scaling. `None` (scaling disabled) when `base_len == max_len`.
    pub fn kappa(&self) -> Option<f64> {
        if self.base_len == self.max_len {
            return None;
        }
        let head_dim = (self.dim / self.n_heads) as f64;
        Some((head_dim.sqrt() - 1.0) / (self.max_len as f64 / self.base_len as f64).ln())
    }
}
