/// TiRex model configuration (fixed from checkpoint hyper_parameters).
#[derive(Clone)]
pub struct TiRexConfig {
    pub patch_size: usize,
    pub num_blocks: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub input_ff_dim: usize,
    pub ffn_up_dim: usize,
    pub train_ctx_len: usize,
    pub quantiles: Vec<f32>,
    pub num_quantiles: usize,
}

impl TiRexConfig {
    pub fn default_from_ckpt() -> Self {
        let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let num_quantiles = quantiles.len();
        Self {
            patch_size: 32,
            num_blocks: 12,
            embedding_dim: 512,
            num_heads: 4,
            input_ff_dim: 2048,
            ffn_up_dim: 1408, // round_up(512 * 2.6667, 64)
            train_ctx_len: 2048,
            quantiles,
            num_quantiles,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embedding_dim / self.num_heads
    }

    pub fn num_patches(&self) -> usize {
        self.train_ctx_len / self.patch_size
    }

    pub fn output_dim(&self) -> usize {
        self.num_quantiles * self.patch_size
    }

    pub fn input_dim(&self) -> usize {
        self.patch_size * 2 // values + mask
    }
}
