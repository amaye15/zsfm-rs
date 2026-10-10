use std::path::PathBuf;

use crate::common::DtypeArg;
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

#[derive(Subcommand)]
pub enum Command {
    /// Download a Mitra variant from HuggingFace and convert to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF
    /// alongside it, deleting the (large) downloaded weight file afterward. Later
    /// conversions (any --dtype) recast from that cached F32 GGUF instead of
    /// re-downloading — pass --redownload to force a fresh download anyway (e.g.
    /// the repo was updated).
    Convert {
        /// `autogluon/mitra-classifier` or `autogluon/mitra-regressor` — pick the one
        /// matching --task (there's no single checkpoint serving both, unlike tabfm).
        #[arg(short, long, default_value = "autogluon/mitra-classifier")]
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
        /// Force a fresh download even if a cached F32 GGUF already exists.
        #[arg(long)]
        redownload: bool,
    },
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
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
    /// Mitra is the only tabular model whose `predict_classification` exposes raw
    /// pre-softmax logits alongside probabilities — tabdpt/tabicl/tabpfn's underlying
    /// model APIs only ever produce probabilities, so their `infer` responses have no
    /// `logits` field.
    Infer {
        #[arg(short, long)]
        gguf: PathBuf,
        #[arg(long, value_enum, default_value = "classification")]
        task: TaskArg,
    },

    /// Delete cached model files for this model.
    ///
    /// Removes the canonical F32 GGUF and config cache at
    /// `<model_dir>/[<variant>-]<owner>__<name>/` created by `convert`.
    /// By default only the cache is removed; pass `--output <path>` to also
    /// delete a converted GGUF file (e.g. `gguf/...`).
    Delete {
        /// HuggingFace repo id (must match the `convert` --model you used).
        #[arg(short, long, default_value = "autogluon/mitra-classifier")]
        model: String,
        /// Which variant to delete (must match `convert` --task).
        #[arg(long, value_enum, default_value = "classification")]
        task: TaskArg,
        /// Directory where the cache was written (must match `convert` --model-dir).
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        /// Also delete this output GGUF file if it exists.
        #[arg(short, long)]
        output: Option<PathBuf>,
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
        Command::Delete {
            model,
            task,
            model_dir,
            output,
        } => {
            let task_str = match task {
                TaskArg::Classification => "classification",
                TaskArg::Regression => "regression",
            };
            let variant_dir = model_dir.join(format!("mitra-{task_str}"));
            let canonical = zsfm_hub::canonical_gguf_path(&variant_dir, &model);
            crate::common::delete_cached_model(&canonical, output.as_deref())?;
        }
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert {
            model,
            task,
            output,
            dtype,
            model_dir,
            token,
            redownload,
        } => {
            let task_str = match task {
                TaskArg::Classification => "classification",
                TaskArg::Regression => "regression",
            };
            let variant_dir = model_dir.join(format!("mitra-{task_str}"));
            let output = output.unwrap_or_else(|| {
                PathBuf::from(format!("gguf/mitra-{task_str}-{}.gguf", dtype_name(&dtype)))
            });
            let canonical = zsfm_hub::canonical_gguf_path(&variant_dir, &model);

            if crate::common::try_recast_from_cache(&canonical, &output, dtype.into(), redownload)?
            {
                return Ok(());
            }

            eprintln!("Downloading {model} into {} …", variant_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &variant_dir)
                .await
                .context("download failed")?;
            let safetensors_path = files
                .safetensors_shards
                .first()
                .context("no safetensors file downloaded")?;

            let config = match task {
                TaskArg::Classification => MitraConfig::classifier(),
                TaskArg::Regression => MitraConfig::regressor(),
            };
            if files.config_json.exists() {
                let config_str =
                    std::fs::read_to_string(&files.config_json).context("read config.json")?;
                let v: serde_json::Value =
                    serde_json::from_str(&config_str).context("parse config.json")?;
                println!(
                    "HF config.json: dim={} n_layers={} n_heads={} dim_output={}",
                    v.get("dim").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("n_layers").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("n_heads").and_then(|x| x.as_u64()).unwrap_or(0),
                    v.get("dim_output").and_then(|x| x.as_u64()).unwrap_or(0),
                );
            }

            let f32_opts = ConvertOptions {
                output_dtype: GGMLType::F32,
            };
            convert(
                std::slice::from_ref(safetensors_path),
                &config,
                &f32_opts,
                &canonical,
            )?;
            eprintln!("Wrote canonical F32 GGUF to {} …", canonical.display());
            files.cleanup_weights();

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            eprintln!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf, task } => {
            if zsfm_burn::engine_from_env() == zsfm_burn::Engine::Burn {
                return run_infer_burn(&gguf, &task).await;
            }
            let buf = zsfm_core::read_stdin_limited()?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let x_support = zsfm_core::parse_matrix(&req["x_support"])?;
            let x_query = zsfm_core::parse_matrix(&req["x_query"])?;

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
                    let n_classes = req
                        .get("n_classes")
                        .and_then(|v| v.as_u64())
                        .unwrap_or_else(|| {
                            y_support
                                .iter()
                                .copied()
                                .max()
                                .map(|m| m as u64 + 1)
                                .unwrap_or(1)
                        }) as usize;

                    eprintln!(
                        "Running classification ({} support / {} query rows, {n_classes} classes) …",
                        x_support.len(),
                        x_query.len()
                    );
                    let logits = model
                        .predict_classification(&x_support, &y_support, &x_query, n_classes)
                        .context("predict")?;
                    let probabilities: Vec<Vec<f32>> =
                        logits.iter().map(|row| zsfm_core::softmax(row)).collect();

                    #[derive(Serialize)]
                    struct Resp<'a> {
                        task: &'static str,
                        logits: &'a [Vec<f32>],
                        probabilities: Vec<Vec<f32>>,
                    }
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&Resp {
                            task: "classification",
                            logits: &logits,
                            probabilities
                        })?
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
                    let predictions = model
                        .predict_regression(&x_support, &y_support, &x_query)
                        .context("predict")?;

                    #[derive(Serialize)]
                    struct Resp {
                        task: &'static str,
                        predictions: Vec<f32>,
                    }
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&Resp {
                            task: "regression",
                            predictions
                        })?
                    );
                }
            }
        }
    }

    Ok(())
}

async fn run_infer_burn(gguf: &std::path::PathBuf, task: &TaskArg) -> anyhow::Result<()> {
    use zsfm_mitra::infer::burn::BurnMitraModel;

    let buf = zsfm_core::read_stdin_limited()?;
    let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

    let x_support = zsfm_core::parse_matrix(&req["x_support"])?;
    let x_query = zsfm_core::parse_matrix(&req["x_query"])?;

    let config = match task {
        TaskArg::Classification => MitraConfig::classifier(),
        TaskArg::Regression => MitraConfig::regressor(),
    };

    eprintln!("Loading Burn model from {} …", gguf.display());
    let model = BurnMitraModel::load(gguf, config).context("load model")?;

    match task {
        TaskArg::Classification => {
            let y_support: Vec<usize> = serde_json::from_value(req["y_support"].clone())
                .context("expected a 1D JSON integer array for `y_support`")?;
            let n_classes = req
                .get("n_classes")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| {
                    y_support
                        .iter()
                        .copied()
                        .max()
                        .map(|m| m as u64 + 1)
                        .unwrap_or(1)
                }) as usize;

            eprintln!(
                "Running classification ({} support / {} query rows, {n_classes} classes) …",
                x_support.len(),
                x_query.len()
            );
            let logits = model
                .predict_classification(&x_support, &y_support, &x_query, n_classes)
                .context("predict")?;
            let probabilities: Vec<Vec<f32>> =
                logits.iter().map(|row| zsfm_core::softmax(row)).collect();

            #[derive(Serialize)]
            struct Resp<'a> {
                task: &'static str,
                logits: &'a [Vec<f32>],
                probabilities: Vec<Vec<f32>>,
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&Resp {
                    task: "classification",
                    logits: &logits,
                    probabilities
                })?
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
            let predictions = model
                .predict_regression(&x_support, &y_support, &x_query)
                .context("predict")?;
            #[derive(Serialize)]
            struct Resp {
                task: &'static str,
                predictions: Vec<f32>,
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&Resp {
                    task: "regression",
                    predictions
                })?
            );
        }
    }
    Ok(())
}
