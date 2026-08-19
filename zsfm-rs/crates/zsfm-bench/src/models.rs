use std::path::Path;

use anyhow::{Context, Result};
use candle_core::Device;

use zsfm_chronos::config::Chronos2Config;
use zsfm_chronos::infer::ChronosModel;
use zsfm_flowstate::config::FlowStateConfig;
use zsfm_flowstate::infer::FlowStateModel;
use zsfm_lag_llama::config::LagLlamaConfig;
use zsfm_lag_llama::infer::LagLlamaModel;
use zsfm_moirai::config::MoiraiConfig;
use zsfm_moirai::infer::MoiraiModel;
use zsfm_moirai2::config::Moirai2Config;
use zsfm_moirai2::infer::Moirai2Model;
use zsfm_moment::config::MomentConfig;
use zsfm_moment::infer::MomentModel;
use zsfm_sundial::infer::SundialModel;
use zsfm_timesfm::infer::TimesFMModel;
use zsfm_tirex::config::TiRexConfig;
use zsfm_tirex::infer::TiRexModel;
use zsfm_toto::infer::TotoModel;
use zsfm_ttm::config::TtmConfig;
use zsfm_ttm::infer::TtmModel;

/// Sundial's ODE step count — 10 Heun steps beats the 50-step default on both
/// speed and MAE (see `benchmark/optimisation_log.md`).
const SUNDIAL_STEPS: usize = 10;
/// Toto's max context window fed to the model, matching the CLI's own default.
const TOTO_MAX_CTX: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelId {
    Toto,
    Chronos,
    TimesFM,
    Sundial,
    Ttm,
    LagLlama,
    Moment,
    Moirai,
    Moirai2,
    FlowState,
    Tirex,
}

impl ModelId {
    pub const ALL: [ModelId; 11] = [
        ModelId::Toto,
        ModelId::Chronos,
        ModelId::TimesFM,
        ModelId::Sundial,
        ModelId::Ttm,
        ModelId::LagLlama,
        ModelId::Moment,
        ModelId::Moirai,
        ModelId::Moirai2,
        ModelId::FlowState,
        ModelId::Tirex,
    ];

    /// CLI-facing name — also what `--models` on `zsfm-bench run` expects.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelId::Toto => "toto",
            ModelId::Chronos => "chronos",
            ModelId::TimesFM => "timesfm",
            ModelId::Sundial => "sundial",
            ModelId::Ttm => "ttm",
            ModelId::LagLlama => "lag_llama",
            ModelId::Moment => "moment",
            ModelId::Moirai => "moirai",
            ModelId::Moirai2 => "moirai2",
            ModelId::FlowState => "flowstate",
            ModelId::Tirex => "tirex",
        }
    }

    pub fn parse(s: &str) -> Option<ModelId> {
        Self::ALL.into_iter().find(|m| m.as_str() == s)
    }

    /// Single-letter abbreviation used in ensemble combo labels (matches the
    /// old Python benchmark's `T`/`C`/`F`/... convention).
    pub fn letter(self) -> char {
        match self {
            ModelId::Toto => 'T',
            ModelId::Chronos => 'C',
            ModelId::TimesFM => 'F',
            ModelId::Sundial => 'S',
            ModelId::Ttm => 'K',
            ModelId::LagLlama => 'L',
            ModelId::Moment => 'M',
            ModelId::Moirai => 'O',
            ModelId::Moirai2 => 'P',
            ModelId::FlowState => 'W',
            ModelId::Tirex => 'X',
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ModelId::Toto => "Toto",
            ModelId::Chronos => "Chronos",
            ModelId::TimesFM => "TimesFM",
            ModelId::Sundial => "Sundial",
            ModelId::Ttm => "TTM",
            ModelId::LagLlama => "Lag-Llama",
            ModelId::Moment => "Moment",
            ModelId::Moirai => "Moirai",
            ModelId::Moirai2 => "Moirai-2",
            ModelId::FlowState => "FlowState",
            ModelId::Tirex => "TiRex",
        }
    }

    /// HuggingFace repo id — must match each model's own `zsfm-cli` `Convert`
    /// default `--model` value, since we read from the canonical F32 GGUF
    /// cache that `zsfm <model> convert` populates at
    /// `<models_dir>/<owner>__<repo>/model-f32.gguf`.
    pub fn repo_id(self) -> &'static str {
        match self {
            ModelId::Toto => "Datadog/Toto-2.0-2.5B",
            ModelId::Chronos => "amazon/chronos-2",
            ModelId::TimesFM => "google/timesfm-2.5-200m-pytorch",
            ModelId::Sundial => "thuml/sundial-base-128m",
            ModelId::Ttm => "ibm-granite/granite-timeseries-ttm-r2",
            ModelId::LagLlama => "time-series-foundation-models/Lag-Llama",
            ModelId::Moment => "AutonLab/MOMENT-1-large",
            ModelId::Moirai => "Salesforce/moirai-1.0-R-large",
            ModelId::Moirai2 => "Salesforce/moirai-2.0-R-small",
            ModelId::FlowState => "ibm-granite/granite-timeseries-flowstate-r1",
            ModelId::Tirex => "NX-AI/TiRex",
        }
    }

    fn needs_config(self) -> bool {
        matches!(
            self,
            ModelId::Toto | ModelId::Chronos | ModelId::FlowState | ModelId::Ttm
        )
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One window's forecast, in whatever form each model's native API produces.
pub struct PointResult {
    pub point: Vec<f32>,
    /// Mean q0.90 − q0.10 width over the horizon, for models that emit
    /// quantiles (used only by the `uncertain` ensemble strategy).
    pub iqr: Option<f32>,
}

pub enum LoadedModel {
    Toto(TotoModel),
    Chronos(ChronosModel),
    TimesFM(TimesFMModel),
    Sundial(SundialModel),
    Ttm(TtmModel),
    LagLlama(LagLlamaModel),
    Moment(MomentModel),
    Moirai(MoiraiModel),
    Moirai2(Moirai2Model),
    FlowState(FlowStateModel),
    Tirex(TiRexModel),
}

fn mean_diff(hi: &[f32], lo: &[f32]) -> f32 {
    let n = hi.len().min(lo.len());
    if n == 0 {
        return f32::NAN;
    }
    let sum: f32 = (0..n).map(|i| hi[i] - lo[i]).sum();
    sum / n as f32
}

fn find_level(levels: &[f32], target: f32) -> Option<usize> {
    levels.iter().position(|&l| (l - target).abs() < 1e-6)
}

impl LoadedModel {
    /// Load the canonical F32 GGUF for `id` from `<models_dir>/<owner>__<repo>/model-f32.gguf`,
    /// the cache `zsfm <model> convert` populates on first conversion. Errors with a clear
    /// "run this command first" message if it doesn't exist yet.
    pub fn load(id: ModelId, models_dir: &Path) -> Result<LoadedModel> {
        let repo_id = id.repo_id();
        let canonical = zsfm_hub::canonical_gguf_path(models_dir, repo_id);
        anyhow::ensure!(
            canonical.exists(),
            "no cached F32 GGUF for {id} at {} — run `zsfm {id} convert` first (any --dtype; \
             the canonical F32 copy is cached automatically)",
            canonical.display()
        );

        let config_path = models_dir
            .join(repo_id.replace('/', "__"))
            .join("config.json");
        if id.needs_config() {
            anyhow::ensure!(
                config_path.exists(),
                "missing {} — should have been cached alongside the GGUF by `zsfm {id} convert`",
                config_path.display()
            );
        }

        let model = match id {
            ModelId::Toto => {
                let config_str = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("read {}", config_path.display()))?;
                let toto_config: serde_json::Value =
                    serde_json::from_str(&config_str).context("parse config.json")?;
                let m = TotoModel::builder(&canonical)
                    .config_json(&toto_config)
                    .with_compute_f64(false)
                    .build()
                    .context("load toto model")?;
                LoadedModel::Toto(m)
            }
            ModelId::Chronos => {
                let config_str = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("read {}", config_path.display()))?;
                let c2_config =
                    Chronos2Config::from_json(&config_str).context("parse config.json")?;
                let m = ChronosModel::builder(&canonical)
                    .config_from(&c2_config)
                    .build()
                    .context("load chronos model")?;
                LoadedModel::Chronos(m)
            }
            ModelId::FlowState => {
                let config_str = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("read {}", config_path.display()))?;
                let fs_config =
                    FlowStateConfig::from_json(&config_str).context("parse config.json")?;
                let m = FlowStateModel::builder(&canonical)
                    .config_from(&fs_config)
                    .build()
                    .context("load flowstate model")?;
                LoadedModel::FlowState(m)
            }
            ModelId::Ttm => {
                let config_str = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("read {}", config_path.display()))?;
                let ttm_config = TtmConfig::from_json(&config_str).context("parse config.json")?;
                let m = TtmModel::builder(&canonical)
                    .config(ttm_config)
                    .build()
                    .context("load ttm model")?;
                LoadedModel::Ttm(m)
            }
            ModelId::TimesFM => {
                let m = TimesFMModel::load(&canonical).context("load timesfm model")?;
                LoadedModel::TimesFM(m)
            }
            ModelId::Sundial => {
                let m = SundialModel::builder(&canonical)
                    .steps(SUNDIAL_STEPS)
                    .build()
                    .context("load sundial model")?;
                LoadedModel::Sundial(m)
            }
            ModelId::LagLlama => {
                let m = LagLlamaModel::load(&canonical, LagLlamaConfig::default_from_ckpt())
                    .context("load lag-llama model")?;
                LoadedModel::LagLlama(m)
            }
            ModelId::Moment => {
                let m = MomentModel::load(&canonical, MomentConfig::default())
                    .context("load moment model")?;
                LoadedModel::Moment(m)
            }
            ModelId::Moirai => {
                let m = MoiraiModel::load(&canonical, MoiraiConfig::default())
                    .context("load moirai model")?;
                LoadedModel::Moirai(m)
            }
            ModelId::Moirai2 => {
                let m = Moirai2Model::load(&canonical, Moirai2Config::default())
                    .context("load moirai2 model")?;
                LoadedModel::Moirai2(m)
            }
            ModelId::Tirex => {
                let m = TiRexModel::load(&canonical, TiRexConfig::default_from_ckpt())
                    .context("load tirex model")?;
                LoadedModel::Tirex(m)
            }
        };
        Ok(model)
    }

    /// Run one window through this model's own native inference path (not the
    /// object-erased `zsfm_core::Forecaster` trait — several models expose
    /// extra info through their native API that the trait doesn't carry,
    /// e.g. TiRex's true `mean` output vs. picking a quantile row).
    /// Replicates each model's own CLI `infer` handler's context
    /// trimming/validation exactly, so results match real `zsfm infer` usage.
    pub fn point_and_iqr(&self, ctx: &[f32], horizon: usize) -> Result<PointResult> {
        match self {
            LoadedModel::Toto(m) => {
                let patch_size = m.config.patch_size();
                let ctx_len = (ctx.len().min(TOTO_MAX_CTX) / patch_size) * patch_size;
                anyhow::ensure!(
                    ctx_len > 0,
                    "context too short for toto (patch_size={patch_size})"
                );
                let start = ctx.len() - ctx_len;
                let data = vec![ctx[start..].to_vec()];
                let mask = vec![vec![true; ctx_len]];
                let qmat = m.forecast(&data, &mask, horizon)?; // [9][1][horizon], q0.1..q0.9
                let point = qmat[4][0].clone();
                let iqr = Some(mean_diff(&qmat[8][0], &qmat[0][0]));
                Ok(PointResult { point, iqr })
            }
            LoadedModel::Chronos(m) => {
                let patch_size = m.config.patch_size();
                let patch_stride = m.config.patch_stride();
                let max_ctx = m.config.context_length();
                let levels = m.config.quantiles().to_vec();
                let median_idx = find_level(&levels, 0.5).unwrap_or(levels.len() / 2);
                let total_len = ctx.len();
                let usable = total_len.min(max_ctx);
                let ctx_len = (usable / patch_stride) * patch_stride;
                anyhow::ensure!(
                    ctx_len >= patch_size,
                    "context too short for chronos (need >= {patch_size}, got {ctx_len})"
                );
                let start = total_len.saturating_sub(ctx_len);
                let qmat = m.forecast(&ctx[start..], horizon)?; // [n_q][horizon]
                let point = qmat[median_idx].clone();
                let iqr = match (find_level(&levels, 0.10), find_level(&levels, 0.90)) {
                    (Some(lo), Some(hi)) => Some(mean_diff(&qmat[hi], &qmat[lo])),
                    _ => None,
                };
                Ok(PointResult { point, iqr })
            }
            LoadedModel::FlowState(m) => {
                let levels = m.config.quantiles().to_vec();
                let median_idx = m.config.median_index();
                let qmat = m.forecast(ctx, horizon)?; // [n_q][horizon]
                let point = qmat[median_idx].clone();
                let iqr = match (find_level(&levels, 0.10), find_level(&levels, 0.90)) {
                    (Some(lo), Some(hi)) => Some(mean_diff(&qmat[hi], &qmat[lo])),
                    _ => None,
                };
                Ok(PointResult { point, iqr })
            }
            LoadedModel::TimesFM(m) => {
                let outputs = m.forecast(ctx, horizon)?; // [10][horizon]: 0=point, 1..9=q0.1..q0.9
                let point = outputs.first().cloned().unwrap_or_default();
                let iqr = if outputs.len() >= 10 {
                    Some(mean_diff(&outputs[9], &outputs[1]))
                } else {
                    None
                };
                Ok(PointResult { point, iqr })
            }
            LoadedModel::Tirex(m) => {
                let (quantiles, mean) = m.forecast(ctx, horizon)?; // [9][horizon] q0.1..q0.9, mean
                let iqr = if quantiles.len() >= 9 {
                    Some(mean_diff(&quantiles[8], &quantiles[0]))
                } else {
                    None
                };
                Ok(PointResult { point: mean, iqr })
            }
            LoadedModel::Sundial(m) => {
                let raw = m.forecast(ctx, &Device::Cpu)?;
                let point = raw.into_iter().take(horizon).collect();
                Ok(PointResult { point, iqr: None })
            }
            LoadedModel::Ttm(m) => {
                let raw = m.forecast(ctx)?;
                let point = raw.into_iter().take(horizon).collect();
                Ok(PointResult { point, iqr: None })
            }
            LoadedModel::LagLlama(m) => Ok(PointResult {
                point: m.forecast(ctx, horizon)?,
                iqr: None,
            }),
            LoadedModel::Moment(m) => Ok(PointResult {
                point: m.forecast(ctx, horizon)?,
                iqr: None,
            }),
            LoadedModel::Moirai(m) => Ok(PointResult {
                point: m.forecast(ctx, horizon)?,
                iqr: None,
            }),
            LoadedModel::Moirai2(m) => Ok(PointResult {
                point: m.forecast(ctx, horizon)?,
                iqr: None,
            }),
        }
    }
}
