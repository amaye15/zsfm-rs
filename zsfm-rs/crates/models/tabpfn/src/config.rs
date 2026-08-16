/// TabPFN-3 (Prior-Labs/tabpfn_3) architecture config — matches the real checkpoint's embedded
/// `config` dict exactly (classifier and regressor share this shape; only `task_type`/heads
/// differ, and only classification is implemented here).
#[derive(Clone, Debug)]
pub struct TabPfnConfig {
    pub embed_dim: usize,
    pub dist_embed_num_blocks: usize,
    pub dist_embed_num_heads: usize,
    pub dist_embed_num_inducing_points: usize,
    pub feature_group_size: usize,
    pub feat_agg_num_blocks: usize,
    pub feat_agg_num_heads: usize,
    pub feat_agg_num_cls_tokens: usize,
    pub nlayers: usize,
    pub icl_num_heads: usize,
    pub icl_num_kv_heads_test: Option<usize>,
    pub decoder_head_dim: usize,
    pub decoder_num_heads: usize,
    pub decoder_use_softmax_scaling: bool,
    pub ff_factor: usize,
    pub softmax_scaling_mlp_hidden_dim: usize,
    pub max_num_classes: usize,
    pub use_nan_indicators: bool,
}

impl TabPfnConfig {
    /// Values from the real `tabpfn-v3-classifier-v3_default.ckpt`'s embedded config.
    pub fn v3_default() -> Self {
        Self {
            embed_dim: 128,
            dist_embed_num_blocks: 3,
            dist_embed_num_heads: 8,
            dist_embed_num_inducing_points: 128,
            feature_group_size: 3,
            feat_agg_num_blocks: 3,
            feat_agg_num_heads: 8,
            feat_agg_num_cls_tokens: 4,
            nlayers: 24,
            icl_num_heads: 8,
            icl_num_kv_heads_test: Some(1),
            decoder_head_dim: 64,
            decoder_num_heads: 6,
            decoder_use_softmax_scaling: true,
            ff_factor: 2,
            softmax_scaling_mlp_hidden_dim: 64,
            max_num_classes: 160,
            use_nan_indicators: true,
        }
    }

    pub fn icl_dim(&self) -> usize {
        self.embed_dim * self.feat_agg_num_cls_tokens
    }

    /// `x_embed`'s input width: grouped raw values, doubled if NaN indicators are concatenated.
    pub fn cell_in_features(&self) -> usize {
        if self.use_nan_indicators {
            self.feature_group_size * 2
        } else {
            self.feature_group_size
        }
    }
}
