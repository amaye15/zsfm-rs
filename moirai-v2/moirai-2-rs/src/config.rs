/// Moirai-2.0-R-small configuration.
#[derive(Clone)]
pub struct Moirai2Config {
    pub d_model: usize,           // 384
    pub n_layers: usize,          // 6
    pub n_heads: usize,           // 6
    pub head_dim: usize,          // 64
    pub d_ff: usize,              // 1024
    pub patch_size: usize,        // 16
    pub num_predict_token: usize, // 4
    pub num_quantiles: usize,     // 9
    pub max_seq_len: usize,       // 512 (in timesteps)
    pub rope_dim: usize,          // 32 (partial_factor=(0.0, 0.5) of head_dim=64)
    pub median_quantile: usize,   // 4 (0-indexed: 0.5 is index 4 of 9 levels)
}

impl Moirai2Config {
    pub fn default() -> Self {
        Self {
            d_model: 384,
            n_layers: 6,
            n_heads: 6,
            head_dim: 64,
            d_ff: 1024,
            patch_size: 16,
            num_predict_token: 4,
            num_quantiles: 9,
            max_seq_len: 512,
            rope_dim: 32,
            median_quantile: 4,
        }
    }

    pub fn max_ctx_tokens(&self) -> usize {
        self.max_seq_len / self.patch_size
    }
}
