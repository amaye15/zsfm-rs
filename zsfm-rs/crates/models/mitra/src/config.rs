/// Mitra (Tab2D) model configuration. Both published variants (classifier, regressor) share
/// the same architecture; only `dim_output` and `task` differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Classification,
    Regression,
}

#[derive(Clone, Debug)]
pub struct MitraConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Max classes the classifier head was trained on (10), or 1 for the regressor.
    pub dim_output: usize,
    pub task: Task,
}

impl MitraConfig {
    /// `autogluon/mitra-classifier`: dim=512, n_layers=12, n_heads=4, dim_output=10.
    pub fn classifier() -> Self {
        Self { dim: 512, n_layers: 12, n_heads: 4, dim_output: 10, task: Task::Classification }
    }

    /// `autogluon/mitra-regressor`: dim=512, n_layers=12, n_heads=4, dim_output=1.
    pub fn regressor() -> Self {
        Self { dim: 512, n_layers: 12, n_heads: 4, dim_output: 1, task: Task::Regression }
    }

    /// Parse the HF `config.json` shipped alongside the weights: `{"dim", "dim_output",
    /// "n_layers", "n_heads", "task"}`.
    pub fn from_json(v: &serde_json::Value) -> anyhow::Result<Self> {
        let task = match v["task"].as_str().unwrap_or("CLASSIFICATION") {
            "REGRESSION" => Task::Regression,
            _ => Task::Classification,
        };
        Ok(Self {
            dim: v["dim"].as_u64().unwrap_or(512) as usize,
            n_layers: v["n_layers"].as_u64().unwrap_or(12) as usize,
            n_heads: v["n_heads"].as_u64().unwrap_or(4) as usize,
            dim_output: v["dim_output"].as_u64().unwrap_or(if task == Task::Regression { 1 } else { 10 }) as usize,
            task,
        })
    }
}
