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

use config::TiRexConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::TiRexModel;

#[derive(Parser)]
#[command(name = "tirex-rs", about = "Download, convert, and run TiRex time-series forecasting")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/tirex-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download TiRex from HuggingFace and convert to GGUF.
    ///
    /// Downloads model.ckpt and reads it directly via candle's pickle reader —
    /// no Python or intermediate extraction step required.
    Convert {
        #[arg(short, long, default_value = "NX-AI/TiRex")]
        model: String,
        #[arg(short, long, default_value = "gguf/tirex-f32.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f32")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },
    /// Run TiRex forecasting from a GGUF file.
    ///
    /// Reads context values as a comma-separated list and outputs JSON forecasts.
    Infer {
        #[arg(short, long, default_value = "gguf/tirex-f32.gguf")]
        gguf: PathBuf,
        /// Comma-separated context values.
        #[arg(long)]
        data: String,
        /// Number of future steps to forecast.
        #[arg(long, default_value = "32")]
        horizon: usize,
        /// Output all quantiles in addition to the median.
        #[arg(long)]
        all_outputs: bool,
    },
}

#[derive(Clone, ValueEnum)]
enum DtypeArg { F32, F16, Q8 }

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => GGMLType::F32,
            DtypeArg::F16 => GGMLType::F16,
            DtypeArg::Q8  => GGMLType::Q8_0,
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
            let ckpt_path = download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config = TiRexConfig::default_from_ckpt();
            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&ckpt_path, &config, &opts, &output)?;
        }

        Command::Infer { gguf, data, horizon, all_outputs } => {
            let context: Vec<f32> = data
                .split(',')
                .map(|s| s.trim().parse::<f32>().context("parse context value"))
                .collect::<anyhow::Result<_>>()?;
            anyhow::ensure!(!context.is_empty(), "context must not be empty");

            let config = TiRexConfig::default_from_ckpt();
            eprintln!("Loading model from {} …", gguf.display());
            let model = TiRexModel::load(&gguf, config.clone()).context("load model")?;

            let (quantiles, mean) = model.forecast(&context, horizon).context("forecast")?;

            let mut q_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
            if all_outputs {
                for (i, q) in config.quantiles.iter().enumerate() {
                    q_map.insert(format!("q{:.1}", q), quantiles[i].clone());
                }
            }

            println!("{}", forecast_json("tirex", context.len(), horizon, vec![(mean, q_map)])?);
        }
    }

    Ok(())
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
