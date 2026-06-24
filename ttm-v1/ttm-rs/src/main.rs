mod config;
mod convert;
mod download;
mod gguf;
mod infer;
mod tensor_map;
mod upload;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use config::TtmConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::TtmModel;

#[derive(Parser)]
#[command(name = "ttm-rs", about = "Convert and run ibm-granite/granite-timeseries-ttm-r2")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/ttm-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download TTM from HuggingFace and convert to GGUF.
    Convert {
        #[arg(short, long, default_value = "ibm-granite/granite-timeseries-ttm-r2")]
        model: String,
        #[arg(short, long, default_value = "gguf/ttm-f16.gguf")]
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
    /// Run TTM forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format.
    Infer {
        /// Path to the GGUF file.
        #[arg(short, long, default_value = "gguf/ttm-f16.gguf")]
        gguf: PathBuf,
        /// Path to config.json (original HuggingFace model).
        #[arg(long, default_value = "models/config.json")]
        config: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum DtypeArg { F32, F16, Bf16, Q8 }

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32  => GGMLType::F32,
            DtypeArg::F16  => GGMLType::F16,
            DtypeArg::Bf16 => GGMLType::BF16,
            DtypeArg::Q8   => GGMLType::Q8_0,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Upload { repo, token } => {
            let crate_root = std::env::current_dir().context("current dir")?;
            let model_root = crate_root.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or(crate_root);
            upload::run(&repo, &token, &model_root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token } => {
            println!("Downloading {model} into {} …", model_dir.display());
            let files = download_model(&model, token.as_deref(), &model_dir)
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
            let contexts = parse_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config)
                .with_context(|| format!("read {}", config.display()))?;
            let ttm_config = TtmConfig::from_json(&config_str).context("parse config.json")?;
            let min_ctx = ttm_config.patch_length;

            eprintln!("Loading model from {} …", gguf.display());
            let model = TtmModel::load(&gguf, ttm_config).context("load model")?;

            let mut fc_choices = Vec::new();
            for ctx in &contexts {
                anyhow::ensure!(
                    ctx.len() >= min_ctx,
                    "Need at least {min_ctx} context values, got {}",
                    ctx.len()
                );
                let raw = model.forecast(ctx).context("forecast")?;
                let point: Vec<f32> = raw.into_iter().take(horizon).collect();
                fc_choices.push((point, BTreeMap::<String, Vec<f32>>::new()));
            }
            let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
            println!("{}", forecast_json("ttm", total_ctx, horizon, fc_choices)?);
        }
    }

    Ok(())
}

fn parse_contexts(val: serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
    match val {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            anyhow::bail!("context must be a non-empty array")
        }
        serde_json::Value::Array(arr) => {
            if arr.first().map(|v| v.is_array()).unwrap_or(false) {
                arr.into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        serde_json::from_value::<Vec<f32>>(v)
                            .with_context(|| format!("context[{i}] must be an array of numbers"))
                    })
                    .collect()
            } else {
                let ctx = serde_json::from_value::<Vec<f32>>(serde_json::Value::Array(arr))
                    .context("context must be a JSON array of numbers")?;
                Ok(vec![ctx])
            }
        }
        _ => anyhow::bail!("context must be a JSON array"),
    }
}

fn forecast_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    fc_choices: Vec<(Vec<f32>, BTreeMap<String, Vec<f32>>)>,
) -> anyhow::Result<String> {
    #[derive(Serialize)]
    struct ForecastResponse {
        id: String,
        object: &'static str,
        created: u64,
        model: String,
        choices: Vec<Choice>,
        usage: Usage,
    }
    #[derive(Serialize)]
    struct Choice {
        index: usize,
        forecast: ForecastOutput,
        finish_reason: &'static str,
    }
    #[derive(Serialize)]
    struct ForecastOutput {
        point: Vec<f32>,
        quantiles: BTreeMap<String, Vec<f32>>,
    }
    #[derive(Serialize)]
    struct Usage {
        context_length: usize,
        forecast_length: usize,
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let resp = ForecastResponse {
        id: format!("forecast-{created:016x}"),
        object: "forecast",
        created,
        model: model_name.to_string(),
        choices: fc_choices.into_iter().enumerate().map(|(i, (point, quantiles))| Choice {
            index: i,
            forecast: ForecastOutput { point, quantiles },
            finish_reason: "stop",
        }).collect(),
        usage: Usage { context_length, forecast_length },
    };

    Ok(serde_json::to_string_pretty(&resp)?)
}
