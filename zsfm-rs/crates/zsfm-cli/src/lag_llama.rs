use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_core::ForecastOutput;
use zsfm_gguf::GGMLType;
use zsfm_lag_llama::config::LagLlamaConfig;
use zsfm_lag_llama::convert::{convert, ConvertOptions};
use zsfm_lag_llama::infer::LagLlamaModel;

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
        #[arg(short, long, default_value = "amaye15/lag-llama-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download Lag-Llama from HuggingFace and convert to GGUF.
    ///
    /// Downloads lag-llama.ckpt and reads it directly via candle's pickle reader —
    /// no Python or intermediate extraction step required.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the downloaded .ckpt
    /// afterward. Later conversions (any --dtype) recast from that cached F32 GGUF
    /// instead of re-downloading — pass --redownload to force a fresh download
    /// anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "time-series-foundation-models/Lag-Llama")]
        model: String,
        #[arg(short, long, default_value = "gguf/lag_llama-f32.gguf")]
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
    /// Print all tensor names in a local safetensors file.
    InspectTensors {
        path: PathBuf,
    },
    /// Run Lag-Llama forecasting from a GGUF file. Univariate, point-forecast only.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format.
    Infer {
        #[arg(short, long, default_value = "gguf/lag_llama-f32.gguf")]
        gguf: PathBuf,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
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
            // download_file() only creates model_dir itself, not the namespaced
            // subdirectory canonical_gguf_path() computes — unlike download_model(),
            // which creates that exact directory as a side effect.
            if let Some(parent) = canonical.parent() {
                std::fs::create_dir_all(parent).context("create canonical GGUF dir")?;
            }

            println!("Downloading {model} into {} …", model_dir.display());
            let ckpt_path = zsfm_hub::download_file(&model, "lag-llama.ckpt", token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config = LagLlamaConfig::default_from_ckpt();
            let f32_opts = ConvertOptions { output_dtype: GGMLType::F32 };
            convert(&ckpt_path, &config, &f32_opts, &canonical)?;
            println!("Wrote canonical F32 GGUF to {} …", canonical.display());
            let _ = std::fs::remove_file(&ckpt_path);

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let tensors = safetensors::SafeTensors::deserialize(&bytes)
                .context("deserialize safetensors")?;
            println!("Tensors in {}:", path.display());
            let mut names: Vec<_> = tensors.names().into_iter().collect();
            names.sort();
            for name in names {
                let t = tensors.tensor(name).unwrap();
                println!("  {name:80} {:?} {:?}", t.dtype(), t.shape());
            }
        }

        Command::Infer { gguf } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;
            let config = LagLlamaConfig::default_from_ckpt();

            eprintln!("Loading model from {} …", gguf.display());
            let model = LagLlamaModel::load(&gguf, config).context("load model")?;

            let mut fc_outputs = Vec::new();
            let mut total_ctx = 0usize;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "Lag-Llama only supports univariate forecasting (1 variate per context)"
                );
                let ctx = &raw_variates[0];
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                total_ctx += ctx.len();
                let point = model.forecast(ctx, horizon).context("forecast")?;
                fc_outputs.push(ForecastOutput::Univariate { point, quantiles: Default::default() });
            }
            println!("{}", zsfm_core::forecast_response_json("lag-llama", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
