use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_flowstate::config::FlowStateConfig;
use zsfm_flowstate::convert::{convert, ConvertOptions};
use zsfm_flowstate::infer::FlowStateModel;
use zsfm_gguf::GGMLType;

#[derive(Subcommand)]
pub enum Command {
    /// Download a FlowState model from HuggingFace and convert it to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
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
        /// Force a fresh download even if a cached F32 GGUF already exists.
        #[arg(long)]
        redownload: bool,
    },
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
    InspectTensors {
        path: PathBuf,
    },
    /// Run FlowState-R1 forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with all quantile levels.
    /// The `point` field contains the median quantile (q0.5). Univariate only.
    Infer {
        #[arg(short, long, default_value = "gguf/flowstate-r1-f16.gguf")]
        gguf: PathBuf,
        #[arg(long, default_value = "models/ibm-granite__granite-timeseries-flowstate-r1/config.json")]
        config: PathBuf,
    },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/flowstate-r1-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
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
            DtypeArg::Q8  => GGMLType::Q8_0,
        }
    }
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token, redownload } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            if canonical.exists() && !redownload {
                println!("Using cached F32 GGUF at {} …", canonical.display());
                zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
                println!("Wrote {}", output.display());
                return Ok(());
            }

            println!("Downloading {model} into {} …", model_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config_str = std::fs::read_to_string(&files.config_json).context("read config.json")?;
            let config = FlowStateConfig::from_json(&config_str).context("parse config.json")?;

            println!(
                "Config: {} encoder layers, embed_dim={}, state_dim={}, decoder_dim={}, {} quantiles",
                config.encoder_num_layers,
                config.embedding_feature_dim,
                config.encoder_state_dim,
                config.decoder_dim,
                config.n_quantiles(),
            );

            let f32_opts = ConvertOptions { output_dtype: GGMLType::F32 };
            convert(&model, &files, &config, &f32_opts, &canonical)?;
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

            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config).with_context(|| format!("read {}", config.display()))?;
            let fs_config = FlowStateConfig::from_json(&config_str).context("parse config.json")?;

            eprintln!("Loading model from {} …", gguf.display());
            let model = FlowStateModel::builder(&gguf).config_from(&fs_config).build().context("load model")?;
            let quantile_levels = model.config.quantiles().to_vec();
            let median_idx = model.config.median_index();

            let mut fc_outputs = Vec::new();
            let mut total_ctx: usize = 0;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "FlowState only supports univariate forecasting (1 variate per context)"
                );
                let ctx = &raw_variates[0];
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                total_ctx += ctx.len();

                eprintln!("Running forecast ({} context steps → {horizon} future steps) …", ctx.len());
                let quantile_mat = model.forecast(ctx, horizon).context("forecast")?;
                let qmat: zsfm_core::QuantileMatrix = quantile_mat.into_iter().map(|row| vec![row]).collect();
                fc_outputs.push(zsfm_core::quantile_matrix_to_output(&qmat, &quantile_levels, median_idx));
            }

            println!("{}", zsfm_core::forecast_response_json("flowstate-r1", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
