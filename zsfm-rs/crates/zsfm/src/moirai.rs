use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_core::{ForecastOutput, VariateForecast};
use zsfm_gguf::GGMLType;
use zsfm_moirai::config::MoiraiConfig;
use zsfm_moirai::convert::{convert, ConvertOptions};
use zsfm_moirai::infer::MoiraiModel;

/// BF16 is deliberately not offered here: candle 0.8's GGUF reader (used by every
/// model's own `infer` loader) can't parse ggml dtype 30, so a bf16-converted file
/// would fail to load right back through this same model's `infer` command. The
/// format-agnostic `zsfm convert`/`zsfm inspect` path supports BF16 for interop with
/// other GGUF consumers; this per-model path only offers dtypes every `infer` here
/// can actually load.
#[derive(Clone, ValueEnum)]
pub enum DtypeArg { F32, F16, Q8 }

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => GGMLType::F32,
            DtypeArg::F16 => GGMLType::F16,
            DtypeArg::Q8  => GGMLType::Q8_0,
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/moirai-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download Moirai-1.0-R-large from HuggingFace and convert to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "Salesforce/moirai-1.0-R-large")]
        model: String,
        #[arg(short, long, default_value = "gguf/moirai-f32.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f32")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Force a fresh download even if a cached F32 GGUF already exists.
        #[arg(long)]
        redownload: bool,
    },
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
    InspectTensors {
        path: PathBuf,
    },
    /// Run Moirai forecasting from a GGUF file. Point-forecast only, channel-independent
    /// across variates. No --config flag: the architecture is fixed (MoiraiConfig::default()),
    /// unlike chronos/flowstate/toto/ttm, which vary by checkpoint and need config.json.
    ///
    /// Reads a JSON request from stdin.
    ///
    /// Univariate batch:
    ///   {"context": [[t0, t1, ...], [t0, t1, ...]], "horizon": N}
    ///
    /// Multivariate batch (channel-independent per variate):
    ///   {"context": [[[v0_t0, v0_t1, ...], [v1_t0, v1_t1, ...]], ...], "horizon": N}
    ///
    /// Univariate output has "point"/"quantiles" in each choice.
    /// Multivariate output has "variates": [{"point":..., "quantiles":...}, ...].
    Infer {
        #[arg(short, long, default_value = "gguf/moirai-f32.gguf")]
        gguf: PathBuf,
    },

    /// Delete cached model files for this model.
    ///
    /// Removes the canonical F32 GGUF and config cache at
    /// `<model_dir>/<owner>__<name>/` created by `convert`. By default only
    /// the cache is removed; pass `--output <path>` to also delete a
    /// converted GGUF file (e.g. `gguf/...`).
    Delete {
        /// HuggingFace repo id (must match the `convert` --model you used).
        #[arg(short, long, default_value = "Salesforce/moirai-1.0-R-large")]
        model: String,
        /// Directory where the cache was written (must match `convert` --model-dir).
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        /// Also delete this output GGUF file if it exists.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Delete { model, model_dir, output } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            crate::common::delete_cached_model(&canonical, output.as_deref())?;
        }
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token, redownload } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            if canonical.exists() && !redownload {
                println!("Using cached F32 GGUF at {} …", canonical.display());
                zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
                println!("Wrote {}", output.display());
                return Ok(());
            }

            println!("Downloading {model} into {} …", model_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config = MoiraiConfig::default();
            let f32_opts = ConvertOptions { output_dtype: GGMLType::F32 };
            convert(&files.safetensors_shards, &config, &f32_opts, &canonical)?;
            println!("Wrote canonical F32 GGUF to {} …", canonical.display());
            files.cleanup_weights();

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;
            let config = MoiraiConfig::default();

            eprintln!("Loading model from {} …", gguf.display());
            let model = MoiraiModel::load(&gguf, config).context("load model")?;

            let mut fc_outputs: Vec<ForecastOutput> = Vec::new();
            let mut total_ctx: usize = 0;

            for raw_variates in &contexts {
                let n_var = raw_variates.len();
                anyhow::ensure!(n_var > 0, "each context must have at least one variate");

                if n_var == 1 {
                    let ctx = &raw_variates[0];
                    anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                    total_ctx += ctx.len();
                    let point = model.forecast(ctx, horizon).context("forecast")?;
                    fc_outputs.push(ForecastOutput::Univariate {
                        point,
                        quantiles: Default::default(),
                    });
                } else {
                    let mut var_forecasts: Vec<VariateForecast> = Vec::with_capacity(n_var);
                    for (vi, ctx) in raw_variates.iter().enumerate() {
                        anyhow::ensure!(!ctx.is_empty(), "variate {vi} context must not be empty");
                        total_ctx += ctx.len();
                        let point = model.forecast(ctx, horizon)
                            .with_context(|| format!("forecast variate {vi}"))?;
                        var_forecasts.push(VariateForecast { point, quantiles: Default::default() });
                    }
                    fc_outputs.push(ForecastOutput::Multivariate { variates: var_forecasts });
                }
            }

            println!("{}", zsfm_core::forecast_response_json("moirai", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
