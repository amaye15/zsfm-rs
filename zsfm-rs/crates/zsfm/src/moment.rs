use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_core::ForecastOutput;
use zsfm_gguf::GGMLType;
use zsfm_moment::config::MomentConfig;
use zsfm_moment::convert::{convert, ConvertOptions};
use zsfm_moment::infer::MomentModel;

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
        #[arg(short, long, default_value = "amaye15/moment-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download MOMENT-1-large from HuggingFace and convert to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "AutonLab/MOMENT-1-large")]
        model: String,
        #[arg(short, long, default_value = "gguf/moment-f32.gguf")]
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
    /// Run MOMENT forecasting from a GGUF file. Univariate, point-forecast only. No
    /// --config flag: the architecture is fixed (MomentConfig::default()), unlike
    /// chronos/flowstate/toto/ttm, which vary by checkpoint and need config.json.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format.
    Infer {
        #[arg(short, long, default_value = "gguf/moment-f32.gguf")]
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
        #[arg(short, long, default_value = "AutonLab/MOMENT-1-large")]
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

            let config = MomentConfig::default();
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
            let config = MomentConfig::default();

            eprintln!("Loading model from {} …", gguf.display());
            let model = MomentModel::load(&gguf, config).context("load model")?;

            let mut fc_outputs = Vec::new();
            let mut total_ctx = 0usize;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "MOMENT only supports univariate forecasting (1 variate per context)"
                );
                let ctx = &raw_variates[0];
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                total_ctx += ctx.len();
                let point = model.forecast(ctx, horizon).context("forecast")?;
                fc_outputs.push(ForecastOutput::Univariate { point, quantiles: Default::default() });
            }
            println!("{}", zsfm_core::forecast_response_json("moment", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
