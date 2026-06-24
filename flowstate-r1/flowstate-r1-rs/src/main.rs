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

use config::FlowStateConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::{FlowStateModel, InferConfig};

#[derive(Parser)]
#[command(name = "flowstate-r1-rs", about = "Convert and run ibm-granite/granite-timeseries-flowstate-r1")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/flowstate-r1-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download a FlowState model from HuggingFace and convert it to GGUF.
    Convert {
        #[arg(short, long, default_value = "ibm-granite/granite-timeseries-flowstate-r1")]
        model: String,
        #[arg(short, long, default_value = "gguf/flowstate-r1-f16.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },
    /// Print all tensor names in a local safetensors or GGUF file.
    InspectTensors {
        path: PathBuf,
    },
    /// Dump the Legendre basis matrix to CSV for comparison with Python numpy.
    DumpBasis {
        #[arg(long, default_value = "models/config.json")]
        config: PathBuf,
        #[arg(short, long, default_value = "24")]
        prediction_length: usize,
    },
    /// Run FlowState univariate forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with all quantile levels.
    /// The `point` field contains the median quantile (q0.5).
    Infer {
        /// Path to the GGUF file.
        #[arg(short, long, default_value = "gguf/flowstate-r1-f16.gguf")]
        gguf: PathBuf,

        /// Path to config.json from the original HuggingFace model.
        #[arg(long, default_value = "models/config.json")]
        config: PathBuf,
    },
}

#[derive(Clone, ValueEnum)]
enum DtypeArg {
    F32,
    F16,
    Q8,
}

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
            let files = download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config_str = std::fs::read_to_string(&files.config_json)
                .context("read config.json")?;
            let config = FlowStateConfig::from_json(&config_str)
                .context("parse config.json")?;

            println!(
                "Config: {} encoder layers, embed_dim={}, state_dim={}, decoder_dim={}, {} quantiles",
                config.encoder_num_layers,
                config.embedding_feature_dim,
                config.encoder_state_dim,
                config.decoder_dim,
                config.n_quantiles(),
            );

            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&model, &files, &config, &opts, &output)?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "gguf" {
                let mut file = std::fs::File::open(&path)
                    .with_context(|| format!("open {}", path.display()))?;
                let content = candle_core::quantized::gguf_file::Content::read(&mut file)
                    .context("read GGUF")?;
                println!("Tensors in {}:", path.display());
                let mut names: Vec<_> = content.tensor_infos.keys().cloned().collect();
                names.sort();
                for name in names {
                    let ti = &content.tensor_infos[&name];
                    println!("  {name:80} {:?} {:?}", ti.ggml_dtype, ti.shape);
                }
            } else {
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
        }

        Command::DumpBasis { config, prediction_length } => {
            let config_str = std::fs::read_to_string(&config)
                .with_context(|| format!("read {}", config.display()))?;
            let fs_config = FlowStateConfig::from_json(&config_str)
                .context("parse config.json")?;
            let decoder_dim = fs_config.decoder_dim as usize;
            let basis_range = fs_config.basis_range();
            let scale = fs_config.decoder_patch_len as f32 / prediction_length as f32;
            let basis = infer::dump_legendre_basis(prediction_length, decoder_dim, basis_range, scale, fs_config.decoder_patch_len as usize);
            let header: Vec<String> = (0..decoder_dim).map(|d| format!("P{d}")).collect();
            println!("{}", header.join(","));
            for row in &basis {
                let vals: Vec<String> = row[..decoder_dim].iter().map(|v| format!("{v:.8}")).collect();
                println!("{}", vals.join(","));
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
            let fs_config = FlowStateConfig::from_json(&config_str)
                .context("parse config.json")?;

            let infer_config = InferConfig {
                num_layers:        fs_config.encoder_num_layers as usize,
                embed_dim:         fs_config.embedding_feature_dim as usize,
                state_dim:         fs_config.encoder_state_dim as usize,
                n_inputs:          fs_config.n_inputs() as usize,
                decoder_dim:       fs_config.decoder_dim as usize,
                decoder_patch_len: fs_config.decoder_patch_len as usize,
                quantiles:         fs_config.quantiles.clone(),
                basis_range:       fs_config.basis_range(),
                context_length:    fs_config.context_length as usize,
                eps:               1e-5,
            };

            let quantile_levels = infer_config.quantiles.clone();
            let median_idx = quantile_levels
                .iter()
                .position(|&q| (q - 0.5).abs() < 1e-6)
                .unwrap_or(quantile_levels.len() / 2);

            eprintln!("Loading model from {} …", gguf.display());
            let model = FlowStateModel::load(&gguf, infer_config)
                .context("load model")?;

            let mut fc_choices = Vec::new();
            for ctx in &contexts {
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                eprintln!("Running forecast ({} context steps → {horizon} future steps) …", ctx.len());
                let quantile_mat = model.forecast(ctx, horizon).context("forecast")?;

                let point = quantile_mat.get(median_idx).cloned().unwrap_or_default();
                let mut quantiles = BTreeMap::new();
                for (i, &level) in quantile_levels.iter().enumerate() {
                    if let Some(q) = quantile_mat.get(i) {
                        quantiles.insert(format!("{level:.2}"), q.clone());
                    }
                }
                fc_choices.push((point, quantiles));
            }
            let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
            println!("{}", forecast_json("flowstate-r1", total_ctx, horizon, fc_choices)?);
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
