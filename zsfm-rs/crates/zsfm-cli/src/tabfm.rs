use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};
use serde::Serialize;

use zsfm_gguf::GGMLType;
use zsfm_tabfm::config::TabFMConfig;
use zsfm_tabfm::convert::{convert, ConvertOptions};
use zsfm_tabfm::ensemble;
use zsfm_tabfm::ensemble::config_gen::NormMethod;
use zsfm_tabfm::ensemble::orchestrate::{self, EnsembleParams};
use zsfm_tabfm::TabFMModel;

#[derive(Clone, ValueEnum)]
pub enum TaskArg {
    Classification,
    Regression,
}

impl TaskArg {
    fn as_str(&self) -> &'static str {
        match self {
            TaskArg::Classification => "classification",
            TaskArg::Regression => "regression",
        }
    }
}

#[derive(Subcommand)]
pub enum Command {
    /// Download one TabFM variant from HuggingFace and convert to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF
    /// alongside it, deleting the (large) downloaded weight file afterward. Later
    /// conversions (any --dtype) recast from that cached F32 GGUF instead of
    /// re-downloading — pass --redownload to force a fresh download anyway (e.g.
    /// the repo was updated).
    Convert {
        #[arg(short, long, default_value = "google/tabfm-1.0.0-pytorch")]
        model: String,
        /// Which HF subfolder / weight variant to convert.
        #[arg(long, value_enum, default_value = "classification")]
        task: TaskArg,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "f16")]
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
    InspectTensors {
        path: PathBuf,
    },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/tabfm-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Run TabFM classification/regression from a GGUF file.
    ///
    /// Reads a JSON request from stdin:
    ///   {"x": [[...]], "y": [...], "train_size": N, "cat_mask": [...], "d": N}
    /// `x` is `[T][H]` (rows x padded feature columns), `y` is `[T]` labels (any finite
    /// placeholder at test-row positions is fine), `train_size` is how many leading rows are
    /// training rows, `cat_mask` (optional, default all-false) marks categorical columns, `d`
    /// (optional, default H) is the actual unpadded feature count.
    /// Outputs JSON: {"task": ..., "logits"|"predictions": [[...]], "probabilities": [[...]]?}
    Infer {
        #[arg(short, long)]
        gguf: PathBuf,
        #[arg(long, default_value = "models/tabfm-classification/google__tabfm-1.0.0-pytorch/classification_config.json")]
        config: PathBuf,
    },
    /// Full sklearn-wrapper-equivalent pipeline: feature scaling, categorical encoding, and
    /// `n_estimators`-member ensembling (bit-compatible with the real `TabFMClassifier`/
    /// `TabFMRegressor`'s default RNG) on top of the raw model.
    ///
    /// Reads a JSON request from stdin:
    ///   {"x_train": [[...]], "y_train": [...], "x_test": [[...]], "cat_mask": [...],
    ///    "n_estimators": 32, "norm_methods": ["none","power"], "class_shift": true,
    ///    "outlier_threshold": 4.0, "softmax_temperature": 0.9, "average_logits": true,
    ///    "random_state": 42, "binary_calibration_method": null|"platt",
    ///    "multiclass_calibration_method": null|"vector", "num_folds_for_cv": 5,
    ///    "enable_nnls": false, "nnls_beta": 0.75, "calibration_lambda": 0.01}
    /// `x_train`/`x_test` cells may be numbers or strings (categorical columns, per `cat_mask`,
    /// are ordinal-encoded internally). Calibration/NNLS are off by default (matching the sklearn
    /// wrapper) and, when enabled, are fit natively via an out-of-fold procedure on `x_train`
    /// — no separate "fit" step needed. Outputs JSON: classification ->
    /// {"task":"classification","probabilities":[[...]],"predicted_labels":[...],"classes":[...]};
    /// regression -> {"task":"regression","predictions":[...]}.
    EnsemblePredict {
        #[arg(short, long)]
        gguf: PathBuf,
        #[arg(long, default_value = "models/tabfm-classification/google__tabfm-1.0.0-pytorch/classification_config.json")]
        config: PathBuf,
        /// Worker threads for the ensemble-member/OOF-fold parallel loops (default: rayon's own
        /// default, i.e. all logical cores, or the `RAYON_NUM_THREADS` env var if set). On some
        /// machines more threads than ~physical-core-count can *hurt* wall time (contention with
        /// Accelerate/BLAS's own internal matmul threading) — benchmark your deployment if
        /// tuning for latency.
        #[arg(long)]
        threads: Option<usize>,
        /// Members per batched forward pass (default: all `n_estimators` in one batch — BLAS is
        /// generally more efficient on one large matmul than many small ones). Smaller values
        /// trade that off against more `rayon`-parallel chunks; benchmark for a given table
        /// size/machine if tuning. Can also be set via the JSON body's `"batch_size"` field.
        #[arg(long)]
        batch_size: Option<usize>,
    },
}

/// BF16 is deliberately not offered here: candle 0.8's GGUF reader (used by every
/// model's own `infer` loader) can't parse ggml dtype 30, so a bf16-converted file
/// would fail to load right back through this same model's `infer` command. The
/// format-agnostic `zsfm convert`/`zsfm inspect` path supports BF16 for interop with
/// other GGUF consumers; this per-model path only offers dtypes every `infer` here
/// can actually load.
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

        Command::Convert { model, task, output, dtype, model_dir, token, redownload } => {
            let task_str = task.as_str();
            let variant_dir = model_dir.join(format!("tabfm-{task_str}"));
            let output = output.unwrap_or_else(|| PathBuf::from(format!("gguf/tabfm-{task_str}-{}.gguf", dtype_name(&dtype))));
            let canonical = zsfm_hub::canonical_gguf_path(&variant_dir, &model);

            if canonical.exists() && !redownload {
                println!("Using cached F32 GGUF at {} …", canonical.display());
                zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
                println!("Wrote {}", output.display());
                return Ok(());
            }

            println!("Downloading {model} ({task_str}) into {} …", variant_dir.display());
            let files = zsfm_hub::download_model_prefixed(&model, task_str, token.as_deref(), &variant_dir)
                .await
                .context("download failed")?;
            let safetensors_path = files
                .safetensors_shards
                .first()
                .context("no safetensors file downloaded")?;

            let config_str = std::fs::read_to_string(&files.config_json).context("read config.json")?;
            let config = TabFMConfig::from_json(&config_str).context("parse config.json")?;
            println!(
                "Config: is_classifier={} embed_dim={} col_blocks={} row_blocks={} icl_blocks={}",
                config.is_classifier, config.embed_dim, config.col_num_blocks, config.row_num_blocks, config.icl_num_blocks,
            );

            let f32_opts = ConvertOptions { output_dtype: GGMLType::F32 };
            convert(&model, safetensors_path, &config, &f32_opts, &canonical)?;
            println!("Wrote canonical F32 GGUF to {} …", canonical.display());
            files.cleanup_weights();

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf, config } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let x = parse_matrix(&req["x"])?;
            let y = parse_vec_f32(&req["y"])?;
            let train_size = req["train_size"].as_u64().context("train_size must be a non-negative integer")? as usize;
            let cat_mask = if req.get("cat_mask").is_some() && !req["cat_mask"].is_null() {
                Some(parse_vec_bool(&req["cat_mask"])?)
            } else {
                None
            };
            let d = req.get("d").and_then(|v| v.as_u64()).map(|v| v as usize);

            let config_str = std::fs::read_to_string(&config).with_context(|| format!("read {}", config.display()))?;
            let tc = TabFMConfig::from_json(&config_str).context("parse config.json")?;

            eprintln!("Loading model from {} …", gguf.display());
            let model = TabFMModel::builder(&gguf).config_from(&tc).build().context("load model")?;

            eprintln!("Running predict ({} rows, train_size={train_size}) …", x.len());
            let out = model
                .predict(&x, &y, train_size, cat_mask.as_deref(), d)
                .context("predict")?;

            println!("{}", predict_json(tc.is_classifier, &out)?);
        }

        Command::EnsemblePredict { gguf, config, threads, batch_size } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let x_train = parse_matrix_value(&req["x_train"])?;
            let x_test = parse_matrix_value(&req["x_test"])?;
            let cat_mask = parse_vec_bool(&req["cat_mask"])?;
            let n_estimators = req.get("n_estimators").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
            let norm_methods = match req.get("norm_methods") {
                Some(serde_json::Value::Array(arr)) => arr
                    .iter()
                    .map(|v| NormMethod::parse(v.as_str().unwrap_or("none")))
                    .collect::<anyhow::Result<Vec<_>>>()?,
                _ => vec![NormMethod::None, NormMethod::Power],
            };
            let class_shift = req.get("class_shift").and_then(|v| v.as_bool()).unwrap_or(true);
            let outlier_threshold = req.get("outlier_threshold").and_then(|v| v.as_f64()).unwrap_or(4.0);
            let softmax_temperature = req.get("softmax_temperature").and_then(|v| v.as_f64()).unwrap_or(0.9);
            let average_logits = req.get("average_logits").and_then(|v| v.as_bool()).unwrap_or(true);
            let random_state = req.get("random_state").and_then(|v| v.as_u64()).unwrap_or(42);
            let binary_calibration =
                req.get("binary_calibration_method").and_then(|v| v.as_str()).map(|s| s == "platt").unwrap_or(false);
            let multiclass_calibration =
                req.get("multiclass_calibration_method").and_then(|v| v.as_str()).map(|s| s == "vector").unwrap_or(false);
            let num_folds_for_cv = req.get("num_folds_for_cv").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let enable_nnls = req.get("enable_nnls").and_then(|v| v.as_bool()).unwrap_or(false);
            let nnls_beta = req.get("nnls_beta").and_then(|v| v.as_f64()).unwrap_or(0.75);
            let calibration_lambda = req.get("calibration_lambda").and_then(|v| v.as_f64()).unwrap_or(1e-2);
            let batch_size = batch_size.or_else(|| req.get("batch_size").and_then(|v| v.as_u64()).map(|v| v as usize));

            let config_str = std::fs::read_to_string(&config).with_context(|| format!("read {}", config.display()))?;
            let tc = TabFMConfig::from_json(&config_str).context("parse config.json")?;

            eprintln!("Loading model from {} …", gguf.display());
            let model = TabFMModel::builder(&gguf).config_from(&tc).build().context("load model")?;

            let params = EnsembleParams::default()
                .with_n_estimators(n_estimators)
                .with_norm_methods(norm_methods)
                .with_class_shift(class_shift)
                .with_outlier_threshold(outlier_threshold)
                .with_softmax_temperature(softmax_temperature)
                .with_average_logits(average_logits)
                .with_random_state(random_state)
                .with_binary_calibration(binary_calibration)
                .with_multiclass_calibration(multiclass_calibration)
                .with_num_folds_for_cv(num_folds_for_cv)
                .with_enable_nnls(enable_nnls)
                .with_nnls_beta(nnls_beta)
                .with_calibration_lambda(calibration_lambda)
                .with_batch_size(batch_size);

            if tc.is_classifier {
                let y_train = req["y_train"].as_array().context("y_train must be an array")?.clone();
                eprintln!(
                    "Running {n_estimators}-member ensemble classification ({} train / {} test rows) …",
                    x_train.len(),
                    x_test.len()
                );
                let out = ensemble::with_thread_pool(threads, || {
                    orchestrate::run_classification(&model, &x_train, &y_train, &x_test, &cat_mask, &params)
                })
                .context("ensemble classification")?;
                #[derive(Serialize)]
                struct Resp {
                    task: &'static str,
                    probabilities: Vec<Vec<f64>>,
                    predicted_labels: Vec<String>,
                    classes: Vec<String>,
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&Resp {
                        task: "classification",
                        probabilities: out.probabilities,
                        predicted_labels: out.predicted_labels,
                        classes: out.classes,
                    })?
                );
            } else {
                let y_train = parse_vec_f64(&req["y_train"])?;
                eprintln!(
                    "Running {n_estimators}-member ensemble regression ({} train / {} test rows) …",
                    x_train.len(),
                    x_test.len()
                );
                let out = ensemble::with_thread_pool(threads, || {
                    orchestrate::run_regression(&model, &x_train, &y_train, &x_test, &cat_mask, &params)
                })
                .context("ensemble regression")?;
                #[derive(Serialize)]
                struct Resp {
                    task: &'static str,
                    predictions: Vec<f64>,
                }
                println!("{}", serde_json::to_string_pretty(&Resp { task: "regression", predictions: out.predictions })?);
            }
        }
    }

    Ok(())
}

fn parse_matrix(val: &serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
    serde_json::from_value(val.clone()).context("expected a 2D JSON array for `x`")
}

fn parse_vec_f32(val: &serde_json::Value) -> anyhow::Result<Vec<f32>> {
    serde_json::from_value(val.clone()).context("expected a 1D JSON array for `y`")
}

fn parse_vec_f64(val: &serde_json::Value) -> anyhow::Result<Vec<f64>> {
    serde_json::from_value(val.clone()).context("expected a 1D JSON numeric array for `y_train`")
}

/// `x_train`/`x_test` matrices for `EnsemblePredict`: cells may be numbers or strings
/// (categorical columns carry raw string levels through `CategoricalOrdinalEncoder`).
fn parse_matrix_value(val: &serde_json::Value) -> anyhow::Result<Vec<Vec<serde_json::Value>>> {
    serde_json::from_value(val.clone()).context("expected a 2D JSON array")
}

fn parse_vec_bool(val: &serde_json::Value) -> anyhow::Result<Vec<bool>> {
    serde_json::from_value(val.clone()).context("expected a 1D JSON boolean array for `cat_mask`")
}

fn predict_json(is_classifier: bool, out: &[Vec<f32>]) -> anyhow::Result<String> {
    #[derive(Serialize)]
    struct ClassificationResponse<'a> {
        task: &'static str,
        logits: &'a [Vec<f32>],
        probabilities: Vec<Vec<f32>>,
    }
    #[derive(Serialize)]
    struct RegressionResponse<'a> {
        task: &'static str,
        predictions: Vec<f32>,
        raw: &'a [Vec<f32>],
    }

    if is_classifier {
        let probabilities: Vec<Vec<f32>> = out.iter().map(|row| softmax(row)).collect();
        Ok(serde_json::to_string_pretty(&ClassificationResponse {
            task: "classification",
            logits: out,
            probabilities,
        })?)
    } else {
        let predictions: Vec<f32> = out.iter().map(|row| row.first().copied().unwrap_or(0.0)).collect();
        Ok(serde_json::to_string_pretty(&RegressionResponse {
            task: "regression",
            predictions,
            raw: out,
        })?)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}
