/// Lag-Llama model configuration (fixed from checkpoint).
#[derive(Clone)]
pub struct LagLlamaConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_per_head: usize,
    pub n_embd: usize,
    pub mlp_hidden: usize,
    pub feature_size: usize,
    pub n_lags: usize,
    pub n_time_feat: usize,
    pub max_context_length: usize,
    pub lags_seq: Vec<usize>,
}

impl LagLlamaConfig {
    /// Values extracted from the lag-llama.ckpt hyper_parameters.
    pub fn default_from_ckpt() -> Self {
        let lags_seq: Vec<usize> = vec![
            0, 7, 8, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30,
            34, 35, 36, 46, 47, 48, 50, 51, 52, 55, 57, 58, 59, 60, 61, 70, 71, 72,
            83, 94, 95, 96, 102, 103, 104, 117, 118, 119, 120, 121, 142, 143, 144,
            154, 155, 156, 166, 167, 168, 177, 178, 179, 180, 181, 334, 335, 336,
            362, 363, 364, 502, 503, 504, 670, 671, 672, 718, 719, 720, 726, 727,
            728, 1090, 1091, 1092,
        ];
        let n_lags = lags_seq.len(); // 84
        let feature_size = 92;
        let n_time_feat = feature_size - n_lags; // 8
        Self {
            n_layer: 8,
            n_head: 9,
            n_embd_per_head: 16,
            n_embd: 144,       // n_head * n_embd_per_head
            mlp_hidden: 512,
            feature_size,
            n_lags,
            n_time_feat,
            max_context_length: 2048,
            lags_seq,
        }
    }
}
