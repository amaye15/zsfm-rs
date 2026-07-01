mod config;
mod convert;
mod download;
mod gguf;
mod infer;
mod tensor_map;
mod upload;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use config::SundialConfig;
use convert::{convert, ConvertOptions};
use gguf::GGMLType;
use infer::SundialModel;

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

#[derive(Parser)]
#[command(name = "sundial-rs", about = "Sundial / Timer v3 GGUF converter and inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/sundial-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download and convert model to GGUF.
    Convert {
        #[arg(short, long, default_value = "thuml/sundial-base-128m")]
        model: String,
        #[arg(short, long, default_value = "gguf/sundial-f16.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
        dtype: DtypeArg,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        #[arg(long, default_value = "models")]
        cache_dir: PathBuf,
    },
    /// Run inference on a GGUF model.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format.
    Infer {
        /// Path to the GGUF file.
        #[arg(long)]
        gguf: PathBuf,
        /// Override the GGUF ODE step count (default: use model metadata, typically 50).
        /// 10–20 steps recommended; latency scales linearly with this value.
        #[arg(long)]
        steps: Option<u32>,
    },
    /// Print tensor names found in a GGUF file.
    InspectTensors {
        #[arg(long)]
        model: PathBuf,
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
            upload::run(&repo, &token, &model_root).await?;
        }

        Command::Convert { model, output, dtype, token, cache_dir } => {
            let output_dtype: GGMLType = dtype.into();

            let files = download::download_model(&model, token.as_deref(), &cache_dir).await?;

            let cfg_bytes = std::fs::read(&files.config_path)
                .with_context(|| format!("read {}", files.config_path.display()))?;
            let config: SundialConfig = serde_json::from_slice(&cfg_bytes)
                .context("parse config.json")?;

            convert(&model, &files, &config, &ConvertOptions { output_dtype }, &output)?;
        }

        Command::Infer { gguf, steps } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
            let contexts = parse_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let device = candle_core::Device::Cpu;

            eprintln!("Loading model from {} …", gguf.display());
            let m = SundialModel::load(&gguf, &device, steps.map(|s| s as usize))?;
            eprintln!("Model loaded.");

            let mut fc_choices = Vec::new();
            for ctx in &contexts {
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                eprintln!("Running forecast (context len = {}) …", ctx.len());
                let raw = m.forecast(ctx, &device)?;
                let point: Vec<f32> = raw.into_iter().take(horizon).collect();
                fc_choices.push((point, BTreeMap::<String, Vec<f32>>::new()));
            }
            let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
            println!("{}", forecast_json("sundial", total_ctx, horizon, fc_choices)?);
        }

        Command::InspectTensors { model } => {
            inspect_tensors(&model)?;
        }
    }
    Ok(())
}

fn inspect_tensors(path: &Path) -> Result<()> {
    use candle_core::quantized::gguf_file;
    use std::fs::File;
    use std::io::BufReader;

    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::new(f);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("read gguf: {}", e))?;

    println!("Metadata:");
    let mut keys: Vec<_> = content.metadata.keys().collect();
    keys.sort();
    for k in &keys {
        println!("  {} = {:?}", k, content.metadata[*k]);
    }

    println!("\nTensors ({}):", content.tensor_infos.len());
    let mut names: Vec<_> = content.tensor_infos.keys().collect();
    names.sort();
    for name in &names {
        let info = &content.tensor_infos[*name];
        println!("  {} {:?} {:?}", name, info.shape, info.ggml_dtype);
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
