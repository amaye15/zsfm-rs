/// Moirai-1.0-R-large configuration.
#[derive(Clone)]
pub struct MoiraiConfig {
    pub d_model: usize,           // 1024
    pub n_layers: usize,          // 24
    pub n_heads: usize,           // 16
    pub head_dim: usize,          // 64
    pub d_ff: usize,              // 2736
    pub max_seq_len: usize,       // 512 (max context length in timesteps)
    pub patch_sizes: Vec<usize>,  // [8, 16, 32, 64, 128]
    pub max_patch_size: usize,    // 128 (output head dimension)
}

impl MoiraiConfig {
    pub fn default() -> Self {
        Self {
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            head_dim: 64,
            d_ff: 2736,
            max_seq_len: 512,
            patch_sizes: vec![8, 16, 32, 64, 128],
            max_patch_size: 128,
        }
    }

    /// Index of a given patch size in patch_sizes.
    pub fn patch_idx(&self, patch_size: usize) -> usize {
        self.patch_sizes.iter().position(|&p| p == patch_size)
            .unwrap_or(2) // default to index 2 = 32
    }
}
