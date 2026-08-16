/// MOMENT-1-large model configuration (T5-large backbone).
#[derive(Clone)]
pub struct MomentConfig {
    pub d_model: usize,         // 1024
    pub n_layers: usize,        // 24
    pub n_heads: usize,         // 16
    pub head_dim: usize,        // 64  (d_kv)
    pub d_ff: usize,            // 2816
    pub seq_len: usize,         // 512 (max context)
    pub patch_len: usize,       // 8
    pub patch_stride: usize,    // 8
    pub num_patches: usize,     // 64  (seq_len / patch_stride)
    pub rel_attn_num_buckets: usize,    // 32
    pub rel_attn_max_distance: usize,   // 128
    pub layer_norm_eps: f64,    // 1e-6
}

impl MomentConfig {
    pub fn default() -> Self {
        let seq_len = 512;
        let patch_len = 8;
        let patch_stride = 8;
        Self {
            d_model: 1024,
            n_layers: 24,
            n_heads: 16,
            head_dim: 64,
            d_ff: 2816,
            seq_len,
            patch_len,
            patch_stride,
            num_patches: seq_len / patch_stride, // 64
            rel_attn_num_buckets: 32,
            rel_attn_max_distance: 128,
            layer_norm_eps: 1e-6,
        }
    }
}
