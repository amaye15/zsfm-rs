use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_core::ForecastOutput;
use zsfm_gguf::GGMLType;
use zsfm_ttm::config::TtmConfig;
use zsfm_ttm::convert::{convert, ConvertOptions};
use zsfm_ttm::infer::TtmModel;

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
        #[arg(short, long, default_value = "amaye15/ttm-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download TTM from HuggingFace and convert to GGUF.
    Convert {
        #[arg(short, long, default_value = "ibm-granite/granite-timeseries-ttm-r2")]
        model: String,
        #[arg(short, long, default_value = "gguf/ttm-f32.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f32")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },
    /// Print all tensor names in a safetensors file.
    InspectTensors { path: PathBuf },
    /// Run TTM forecasting from a GGUF file. Univariate, point-forecast only.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format.
    Infer {
        /// Path to the GGUF file.
        #[arg(short, long, default_value = "gguf/ttm-f32.gguf")]
        gguf: PathBuf,
        /// Path to config.json (original HuggingFace model).
        #[arg(long, default_value = "models/config.json")]
        config: PathBuf,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token } => {
            println!("Downloading {model} into {} …", model_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config_str = std::fs::read_to_string(&files.config_json)
                .context("read config.json")?;
            let config = TtmConfig::from_json(&config_str).context("parse config.json")?;

            println!(
                "Config: context={}, pred={}, patch={}/{}, d_model={}, {} enc layers, {} dec layers, {} adaptive levels",
                config.context_length, config.prediction_length,
                config.patch_length, config.patch_stride,
                config.d_model, config.num_layers, config.decoder_num_layers,
                config.adaptive_patching_levels,
            );

            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&model, &files, &config, &opts, &output)?;
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

        Command::Infer { gguf, config } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config)
                .with_context(|| format!("read {}", config.display()))?;
            let ttm_config = TtmConfig::from_json(&config_str).context("parse config.json")?;
            let min_ctx = ttm_config.patch_length;

            eprintln!("Loading model from {} …", gguf.display());
            let model = TtmModel::builder(&gguf).config(ttm_config).build().context("load model")?;

            let mut fc_outputs = Vec::new();
            let mut total_ctx = 0usize;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "TTM only supports univariate forecasting (1 variate per context)"
                );
                let ctx = &raw_variates[0];
                anyhow::ensure!(
                    ctx.len() >= min_ctx,
                    "Need at least {min_ctx} context values, got {}",
                    ctx.len()
                );
                total_ctx += ctx.len();
                let raw = model.forecast(ctx).context("forecast")?;
                let point: Vec<f32> = raw.into_iter().take(horizon).collect();
                fc_outputs.push(ForecastOutput::Univariate { point, quantiles: Default::default() });
            }
            println!("{}", zsfm_core::forecast_response_json("ttm", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
