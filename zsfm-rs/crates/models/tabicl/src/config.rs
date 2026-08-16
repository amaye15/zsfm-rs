/// TabICL v2 configuration (`jingang/TabICL`'s `tabicl-classifier-v2-*.ckpt`). Matches the
/// checkpoint's embedded `config` dict exactly. Classification only — regression's
/// quantile-distribution head and the >10-class mixed-radix/hierarchical paths are out of
/// scope (see crate docs).
#[derive(Clone, Debug)]
pub struct TabIclConfig {
    pub max_classes: usize,     // 10
    pub embed_dim: usize,       // 128
    pub col_num_blocks: usize,  // 3
    pub col_nhead: usize,       // 8
    pub col_num_inds: usize,    // 128
    pub feature_group_size: usize, // 3
    pub row_num_blocks: usize,  // 3
    pub row_nhead: usize,       // 8
    pub row_num_cls: usize,     // 4
    pub row_rope_base: f64,     // 100000
    pub icl_num_blocks: usize,  // 12
    pub icl_nhead: usize,       // 8
    pub ff_factor: usize,       // 2
}

impl TabIclConfig {
    pub fn v2() -> Self {
        Self {
            max_classes: 10,
            embed_dim: 128,
            col_num_blocks: 3,
            col_nhead: 8,
            col_num_inds: 128,
            feature_group_size: 3,
            row_num_blocks: 3,
            row_nhead: 8,
            row_num_cls: 4,
            row_rope_base: 100_000.0,
            icl_num_blocks: 12,
            icl_nhead: 8,
            ff_factor: 2,
        }
    }

    pub fn icl_dim(&self) -> usize {
        self.embed_dim * self.row_num_cls
    }
}
