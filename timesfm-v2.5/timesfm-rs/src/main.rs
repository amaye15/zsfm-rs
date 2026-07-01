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

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use config::TimesFMConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::TimesFMModel;

#[derive(Parser)]
#[command(name = "timesfm-rs", about = "TimesFM 2.5 200M → GGUF converter and inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
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

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/timesfm-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download and convert model weights to GGUF format.
    Convert {
        #[arg(long, default_value = "google/timesfm-2.5-200m-pytorch")]
        model: String,
        #[arg(long, default_value = "gguf/timesfm.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
        dtype: DtypeArg,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Local directory to cache downloaded files.
        #[arg(long, default_value = "models")]
        cache_dir: PathBuf,
    },
    /// Inspect tensors stored in a GGUF file.
    InspectTensors {
        #[arg()]
        gguf: PathBuf,
    },
    /// Run inference on a time series.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with point and quantile forecasts.
    Infer {
        #[arg(long)]
        gguf: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Upload { repo, token } => {
            let crate_root = std::env::current_dir().context("current dir")?;
            let model_root = crate_root.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or(crate_root);
            upload::run(&repo, &token, &model_root).await
        }

        Command::Convert { model, output, dtype, token, cache_dir } => {
            cmd_convert(&model, &output, dtype.into(), token.as_deref(), &cache_dir).await
        }
        Command::InspectTensors { gguf } => cmd_inspect(&gguf),
        Command::Infer { gguf } => cmd_infer(&gguf),
    }
}

async fn cmd_convert(
    model: &str,
    output: &PathBuf,
    output_dtype: GGMLType,
    token: Option<&str>,
    cache_dir: &PathBuf,
) -> Result<()> {
    let files = download_model(model, token, cache_dir).await?;
    let config = TimesFMConfig::new();

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {}", parent.display()))?;
        }
    }

    convert(model, &files, &config, &ConvertOptions { output_dtype }, output)
}

fn cmd_inspect(gguf_path: &PathBuf) -> Result<()> {
    use candle_core::quantized::gguf_file;
    use std::io::BufReader;

    let file = std::fs::File::open(gguf_path)
        .with_context(|| format!("open {}", gguf_path.display()))?;
    let mut reader = BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader).context("parse GGUF")?;

    println!("Metadata:");
    for (k, v) in &content.metadata {
        println!("  {k} = {v:?}");
    }

    println!("\nTensors ({}):", content.tensor_infos.len());
    let mut names: Vec<_> = content.tensor_infos.keys().collect();
    names.sort();
    for name in names {
        let info = &content.tensor_infos[name];
        println!("  {name}  shape={:?}  dtype={:?}", info.shape, info.ggml_dtype);
    }
    Ok(())
}

fn cmd_infer(gguf_path: &PathBuf) -> Result<()> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
    let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
    let contexts = parse_contexts(req["context"].clone())?;
    let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

    eprintln!("Loading model from {} …", gguf_path.display());
    let model = TimesFMModel::load(gguf_path)?;

    let quantile_labels = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"];

    let mut fc_choices = Vec::new();
    for ctx in &contexts {
        if ctx.is_empty() {
            bail!("context series must not be empty");
        }
        eprintln!("Running forecast (context={}, horizon={}) …", ctx.len(), horizon);
        // outputs[0] = point forecast, outputs[1..9] = quantile forecasts q0.1..q0.9
        let outputs = model.forecast(ctx, horizon)?;

        let point = outputs.first().cloned().unwrap_or_default();
        let mut quantiles = BTreeMap::new();
        for (i, label) in quantile_labels.iter().enumerate() {
            if let Some(q) = outputs.get(i + 1) {
                quantiles.insert(label.to_string(), q.clone());
            }
        }
        fc_choices.push((point, quantiles));
    }
    let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
    println!("{}", forecast_json("timesfm", total_ctx, horizon, fc_choices)?);
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
