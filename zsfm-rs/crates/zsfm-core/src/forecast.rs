use std::path::Path;

use anyhow::Result;

/// `[quantile_level][variate][horizon_step]` — the output shape every
/// zsfm time-series forecaster produces.
pub type QuantileMatrix = Vec<Vec<Vec<f32>>>;

/// Common interface implemented by every GGUF-quantized zero-shot time-series
/// forecaster in the workspace (Toto, Chronos, TimesFM, …). Lets the CLI and
/// Python bindings dispatch across models without per-model glue code.
pub trait Forecaster: Sized {
    type Config;

    fn load(gguf_path: &Path, config: Self::Config) -> Result<Self>;

    /// `context[variate][timestep]`, `mask[variate][timestep]` (`true` = observed).
    /// Returns `quantiles[quantile_level][variate][horizon_step]`.
    fn forecast(
        &self,
        context: &[Vec<f32>],
        mask: &[Vec<bool>],
        horizon: usize,
    ) -> Result<QuantileMatrix>;
}
