use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Subcommand, ValueEnum};

use zsfm_core::ForecastOutput;
use zsfm_gguf::GGMLType;
use zsfm_timesfm::config::TimesFMConfig;
use zsfm_timesfm::convert::{convert, ConvertOptions};
use zsfm_timesfm::infer::TimesFMModel;

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
        #[arg(short, long, default_value = "amaye15/timesfm-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download and convert model weights to GGUF format.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "google/timesfm-2.5-200m-pytorch")]
        model: String,
        #[arg(short, long, default_value = "gguf/timesfm.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
        dtype: DtypeArg,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
        /// Local directory to cache downloaded files.
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        /// Force a fresh download even if a cached F32 GGUF already exists.
        #[arg(long)]
        redownload: bool,
    },
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
    InspectTensors {
        path: PathBuf,
    },
    /// Run inference on a time series. Univariate only; the architecture is fixed, so no
    /// `--config` is needed.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with point and quantile forecasts.
    Infer {
        #[arg(short, long, default_value = "gguf/timesfm.gguf")]
        gguf: PathBuf,
    },
}

pub async fn run(command: Command) -> Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }
        Command::Convert { model, output, dtype, token, model_dir, redownload } => {
            cmd_convert(&model, &output, dtype.into(), token.as_deref(), &model_dir, redownload).await?;
        }
        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,
        Command::Infer { gguf } => cmd_infer(&gguf)?,
    }
    Ok(())
}

async fn cmd_convert(
    model: &str,
    output: &PathBuf,
    output_dtype: GGMLType,
    token: Option<&str>,
    model_dir: &PathBuf,
    redownload: bool,
) -> Result<()> {
    let canonical = zsfm_hub::canonical_gguf_path(model_dir, model);
    if canonical.exists() && !redownload {
        println!("Using cached F32 GGUF at {} …", canonical.display());
        zsfm_checkpoint::recast(&canonical, output, output_dtype)?;
        println!("Wrote {}", output.display());
        return Ok(());
    }

    println!("Downloading {model} into {} …", model_dir.display());
    let files = zsfm_hub::download_model(model, token, model_dir)
        .await
        .context("download failed")?;
    let config = TimesFMConfig::new();

    convert(model, &files, &config, &ConvertOptions { output_dtype: GGMLType::F32 }, &canonical)?;
    println!("Wrote canonical F32 GGUF to {} …", canonical.display());
    files.cleanup_weights();

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create output dir {}", parent.display()))?;
        }
    }
    zsfm_checkpoint::recast(&canonical, output, output_dtype)?;
    println!("Wrote {}", output.display());
    Ok(())
}

fn cmd_infer(gguf_path: &PathBuf) -> Result<()> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
    let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
    let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
    let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

    eprintln!("Loading model from {} …", gguf_path.display());
    let model = TimesFMModel::load(gguf_path).context("load model")?;

    let quantile_labels = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"];

    let mut fc_outputs = Vec::new();
    let mut total_ctx = 0usize;
    for raw_variates in &contexts {
        anyhow::ensure!(
            raw_variates.len() == 1,
            "TimesFM only supports univariate forecasting (1 variate per context)"
        );
        let ctx = &raw_variates[0];
        if ctx.is_empty() {
            bail!("context series must not be empty");
        }
        total_ctx += ctx.len();
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
        fc_outputs.push(ForecastOutput::Univariate { point, quantiles });
    }
    println!("{}", zsfm_core::forecast_response_json("timesfm", total_ctx, horizon, fc_outputs)?);
    Ok(())
}
