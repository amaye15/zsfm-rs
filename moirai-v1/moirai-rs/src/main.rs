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

use config::MoiraiConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::MoiraiModel;

#[derive(Parser)]
#[command(name = "moirai-rs", about = "Download, convert, and run Moirai-1.0-R-large time-series forecasting")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/moirai-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download Moirai-1.0-R-large from HuggingFace and convert to GGUF.
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
    },
    /// Print all tensor names in a local safetensors file.
    InspectTensors {
        path: PathBuf,
    },
    /// Run Moirai forecasting from a GGUF file.
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

            let config = MoiraiConfig::default();
            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&files.safetensors_shards, &config, &opts, &output)?;
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
            let contexts = parse_mv_contexts(req["context"].clone())?;
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
                        quantiles: BTreeMap::new(),
                    });
                } else {
                    let mut var_forecasts: Vec<VariateForecast> = Vec::with_capacity(n_var);
                    for (vi, ctx) in raw_variates.iter().enumerate() {
                        anyhow::ensure!(!ctx.is_empty(), "variate {vi} context must not be empty");
                        total_ctx += ctx.len();
                        let point = model.forecast(ctx, horizon)
                            .with_context(|| format!("forecast variate {vi}"))?;
                        var_forecasts.push(VariateForecast { point, quantiles: BTreeMap::new() });
                    }
                    fc_outputs.push(ForecastOutput::Multivariate { variates: var_forecasts });
                }
            }

            println!("{}", forecast_json("moirai", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(untagged)]
enum ForecastOutput {
    Univariate {
        point: Vec<f32>,
        quantiles: BTreeMap<String, Vec<f32>>,
    },
    Multivariate {
        variates: Vec<VariateForecast>,
    },
}

#[derive(Serialize)]
struct VariateForecast {
    point: Vec<f32>,
    quantiles: BTreeMap<String, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// JSON input parsing
// ---------------------------------------------------------------------------

/// Parse the `context` field into [batch][variate][time].
///
/// Accepted shapes:
///   [t0, t1, ...]                    → batch=1, n_var=1 (flat single series)
///   [[t0, t1, ...], ...]             → batch=N, n_var=1 (batch of univariate)
///   [[[v0_t0, ...], [v1_t0, ...]], ...] → batch=N, n_var=M (batch of multivariate)
fn parse_mv_contexts(val: serde_json::Value) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    match val {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            anyhow::bail!("context must be a non-empty array")
        }
        serde_json::Value::Array(arr) => {
            let first = arr.first().unwrap();
            if !first.is_array() {
                // Flat: [t0, t1, ...] → batch=1, n_var=1
                let ctx = serde_json::from_value::<Vec<f32>>(serde_json::Value::Array(arr))
                    .context("context must be a JSON array of numbers")?;
                Ok(vec![vec![ctx]])
            } else if first
                .as_array()
                .and_then(|a| a.first())
                .map(|v| v.is_array())
                .unwrap_or(false)
            {
                // 3D: [batch][variate][time]
                arr.into_iter()
                    .enumerate()
                    .map(|(i, batch_item)| {
                        serde_json::from_value::<Vec<Vec<f32>>>(batch_item)
                            .with_context(|| format!("context[{i}] must be an array of variate arrays"))
                    })
                    .collect()
            } else {
                // 2D: [batch][time] → each series is n_var=1
                arr.into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let series = serde_json::from_value::<Vec<f32>>(v)
                            .with_context(|| format!("context[{i}] must be an array of numbers"))?;
                        Ok(vec![series])
                    })
                    .collect()
            }
        }
        _ => anyhow::bail!("context must be a JSON array"),
    }
}

// ---------------------------------------------------------------------------
// JSON response serialisation
// ---------------------------------------------------------------------------

fn forecast_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    fc_outputs: Vec<ForecastOutput>,
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
        choices: fc_outputs.into_iter().enumerate().map(|(i, forecast)| Choice {
            index: i,
            forecast,
            finish_reason: "stop",
        }).collect(),
        usage: Usage { context_length: context_length, forecast_length },
    };

    Ok(serde_json::to_string_pretty(&resp)?)
}
