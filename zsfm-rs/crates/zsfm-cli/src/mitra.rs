use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};
use serde::Serialize;

use zsfm_gguf::GGMLType;
use zsfm_mitra::config::MitraConfig;
use zsfm_mitra::convert::{convert, ConvertOptions};
use zsfm_mitra::MitraModel;

#[derive(Clone, ValueEnum)]
pub enum TaskArg {
    Classification,
    Regression,
}

/// BF16 is deliberately not offered here — see the same note on every other model's CLI:
/// candle 0.8's GGUF reader can't parse ggml dtype 30, so a bf16-converted file couldn't be
/// loaded back by this crate's own `infer` command.
#[derive(Clone, ValueEnum)]
pub enum DtypeArg {
    F32,
    F16,
    Q8,
}

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => GGMLType::F32,
            DtypeArg::F16 => GGMLType::F16,
            DtypeArg::Q8 => GGMLType::Q8_0,
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Download a Mitra variant from HuggingFace and convert to GGUF.
    Convert {
        /// `autogluon/mitra-classifier` or `autogluon/mitra-regressor`.
        #[arg(short, long)]
        model: String,
        #[arg(long, value_enum, default_value = "classification")]
        task: TaskArg,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "f32")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },
    /// Print all tensor names in a local safetensors file.
    InspectTensors { path: PathBuf },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/mitra-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Zero-shot classification/regression from a GGUF file (no fine-tuning — see the crate's
    /// module docs for why).
    ///
    /// Reads a JSON request from stdin:
    ///   {"x_support": [[...]], "y_support": [...], "x_query": [[...]], "n_classes": N}
    /// `n_classes` is only used for classification (ignored, may be omitted, for regression).
    /// Outputs JSON: classification -> {"task":"classification","logits":[[...]],
    /// "probabilities":[[...]]}; regression -> {"task":"regression","predictions":[...]}.
    Infer {
        #[arg(short, long)]
        gguf: PathBuf,
        #[arg(long, value_enum, default_value = "classification")]
        task: TaskArg,
    },
}

fn dtype_name(d: &DtypeArg) -> &'static str {
    match d {
        DtypeArg::F32 => "f32",
        DtypeArg::F16 => "f16",
        DtypeArg::Q8 => "q8",
    }
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, task, output, dtype, model_dir, token } => {
            let task_str = match task {
                TaskArg::Classification => "classification",
                TaskArg::Regression => "regression",
            };
            let variant_dir = model_dir.join(format!("mitra-{task_str}"));
            println!("Downloading {model} into {} …", variant_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &variant_dir)
                .await
                .context("download failed")?;
            let safetensors_path =
                files.safetensors_shards.first().context("no safetensors file downloaded")?;

            let config = match task {
                TaskArg::Classification => MitraConfig::classifier(),
                TaskArg::Regression => MitraConfig::regressor(),
            };
            if files.config_json.exists() {
                let config_str = std::fs::read_to_string(&files.config_json).context("read config.json")?;
                let v: serde_json::Value = serde_json::from_str(&config_str).context("parse config.json")?;
                println!(
                    "HF config.json: dim={} n_layers={} n_heads={} dim_output={}",
                    v.get("dim").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("n_layers").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("n_heads").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("dim_output").and_then(|x| x.as_u64()).unwrap_or(0),
                );
            }

            let output =
                output.unwrap_or_else(|| PathBuf::from(format!("gguf/mitra-{task_str}-{}.gguf", dtype_name(&dtype))));
            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(std::slice::from_ref(safetensors_path), &config, &opts, &output)?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => {
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let tensors = safetensors::SafeTensors::deserialize(&bytes).context("deserialize safetensors")?;
            println!("Tensors in {}:", path.display());
            let mut names: Vec<_> = tensors.names().into_iter().collect();
            names.sort();
            for name in names {
                let t = tensors.tensor(name).unwrap();
                println!("  {name:80} {:?} {:?}", t.dtype(), t.shape());
            }
        }

        Command::Infer { gguf, task } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let x_support = parse_matrix(&req["x_support"])?;
            let x_query = parse_matrix(&req["x_query"])?;

            let config = match task {
                TaskArg::Classification => MitraConfig::classifier(),
                TaskArg::Regression => MitraConfig::regressor(),
            };

            eprintln!("Loading model from {} …", gguf.display());
            let model = MitraModel::load(&gguf, config).context("load model")?;

            match task {
                TaskArg::Classification => {
                    let y_support: Vec<usize> = serde_json::from_value(req["y_support"].clone())
                        .context("expected a 1D JSON integer array for `y_support`")?;
                    let n_classes = req.get("n_classes").and_then(|v| v.as_u64()).unwrap_or_else(|| {
                        y_support.iter().copied().max().map(|m| m as u64 + 1).unwrap_or(1)
                    }) as usize;

                    eprintln!(
                        "Running classification ({} support / {} query rows, {n_classes} classes) …",
                        x_support.len(),
                        x_query.len()
                    );
                    let logits = model
                        .predict_classification(&x_support, &y_support, &x_query, n_classes)
                        .context("predict")?;
                    let probabilities: Vec<Vec<f32>> = logits.iter().map(|row| softmax(row)).collect();

                    #[derive(Serialize)]
                    struct Resp<'a> {
                        task: &'static str,
                        logits: &'a [Vec<f32>],
                        probabilities: Vec<Vec<f32>>,
                    }
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&Resp { task: "classification", logits: &logits, probabilities })?
                    );
                }
                TaskArg::Regression => {
                    let y_support: Vec<f32> = serde_json::from_value(req["y_support"].clone())
                        .context("expected a 1D JSON numeric array for `y_support`")?;

                    eprintln!(
                        "Running regression ({} support / {} query rows) …",
                        x_support.len(),
                        x_query.len()
                    );
                    let predictions = model.predict_regression(&x_support, &y_support, &x_query).context("predict")?;

                    #[derive(Serialize)]
                    struct Resp {
                        task: &'static str,
                        predictions: Vec<f32>,
                    }
                    println!("{}", serde_json::to_string_pretty(&Resp { task: "regression", predictions })?);
                }
            }
        }
    }

    Ok(())
}

fn parse_matrix(val: &serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
    serde_json::from_value(val.clone()).context("expected a 2D JSON array")
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}
