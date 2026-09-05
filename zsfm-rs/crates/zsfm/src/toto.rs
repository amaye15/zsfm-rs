use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_gguf::GGMLType;
use zsfm_toto::config::TotoConfig;
use zsfm_toto::convert::{convert, ConvertOptions};
use zsfm_toto::infer::TotoModel;

#[derive(Subcommand)]
pub enum Command {
    /// Download a Toto model from HuggingFace and convert it to GGUF.
    ///
    /// The first conversion downloads the model and writes a canonical F32 GGUF to
    /// `<model_dir>/<owner>__<name>/model-f32.gguf`, deleting the (large) downloaded
    /// weight files afterward. Later conversions (any --dtype) recast from that
    /// cached F32 GGUF instead of re-downloading — pass --redownload to force a
    /// fresh download anyway (e.g. the repo was updated).
    Convert {
        #[arg(short, long, default_value = "Datadog/Toto-2.0-2.5B")]
        model: String,
        #[arg(short, long, default_value = "gguf/toto-2.5b-f16.gguf")]
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
    /// Run Toto-2 time-series forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin.
    ///
    /// Univariate batch:
    ///   {"context": [[t0, t1, ...], [t0, t1, ...]], "horizon": N}
    ///
    /// Multivariate batch (n_var variates per series):
    ///   {"context": [[[v0_t0, v0_t1, ...], [v1_t0, v1_t1, ...]], ...], "horizon": N}
    Infer {
        /// Path to the GGUF file.
        #[arg(short, long, default_value = "gguf/toto-2.5b-f16.gguf")]
        gguf: PathBuf,

        /// Path to config.json from the original HuggingFace model.
        #[arg(long, default_value = "models/Datadog__Toto-2.0-2.5B/config.json")]
        config: PathBuf,

        /// Context length fed to the model (must be divisible by patch_size=32).
        /// Defaults to the last 4096 timesteps (or all if shorter).
        #[arg(long)]
        context_length: Option<usize>,

        /// Run the forward pass in F64 (double precision) to match PyTorch
        /// numerical accuracy. Uses ~2× memory. Default: F32.
        #[arg(long = "f64", default_value_t = false)]
        use_f64: bool,
    },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/toto-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Delete cached model files for this model.
    ///
    /// Removes the canonical F32 GGUF and config cache at
    /// `<model_dir>/<owner>__<name>/` created by `convert`. By default only
    /// the cache is removed; pass `--output <path>` to also delete a
    /// converted GGUF file (e.g. `gguf/...`).
    Delete {
        /// HuggingFace repo id (must match the `convert` --model you used).
        #[arg(short, long, default_value = "Datadog/Toto-2.0-2.5B")]
        model: String,
        /// Directory where the cache was written (must match `convert` --model-dir).
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        /// Also delete this output GGUF file if it exists.
        #[arg(short, long)]
        output: Option<PathBuf>,
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

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Delete { model, model_dir, output } => {
            let canonical = zsfm_hub::canonical_gguf_path(&model_dir, &model);
            crate::common::delete_cached_model(&canonical, output.as_deref())?;
        }
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

            let config_str = std::fs::read_to_string(&files.config_json)
                .context("read config.json")?;
            let config = TotoConfig::from_json(&config_str)
                .context("parse config.json")?;

            println!(
                "Config: {} layers, hidden={}, heads={}",
                config.num_hidden_layers, config.hidden_size, config.num_attention_heads
            );

            let f32_opts = ConvertOptions { output_dtype: GGMLType::F32 };
            convert(&model, &files, &config, &f32_opts, &canonical)?;
            println!("Wrote canonical F32 GGUF to {} …", canonical.display());
            files.cleanup_weights();

            zsfm_checkpoint::recast(&canonical, &output, dtype.into())?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf, config, context_length, use_f64 } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            // contexts: [batch][variate][time] — univariate is [batch][1][time]
            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config)
                .with_context(|| format!("read {}", config.display()))?;
            let toto_config: serde_json::Value = serde_json::from_str(&config_str)?;

            let max_ctx = context_length.unwrap_or(4096);
            let quantile_levels = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
            let median_idx = 4; // q0.5 is at index 4

            eprintln!("Loading model from {} …", gguf.display());
            let model = TotoModel::builder(&gguf)
                .config_json(&toto_config)
                .with_compute_f64(use_f64)
                .build()
                .context("load model")?;
            let patch_size = model.config.patch_size();

            let mut fc_outputs = Vec::new();
            let mut total_ctx: usize = 0;

            for raw_variates in &contexts {
                let n_var = raw_variates.len();
                anyhow::ensure!(n_var > 0, "each context must have at least one variate");
                // Raw (pre-trim) length, matching every other model's `usage.context_length`
                // convention — how much context the client sent, not how much survived
                // patch-size/max_ctx trimming below.
                total_ctx += raw_variates.iter().map(|v| v.len()).sum::<usize>();

                // Determine patch-aligned ctx_len from the first variate, apply to all
                let v0_len = raw_variates[0].len();
                let ctx_len = (v0_len.min(max_ctx) / patch_size) * patch_size;
                anyhow::ensure!(
                    ctx_len > 0,
                    "context too short — need at least {patch_size} timesteps, got {v0_len}"
                );

                let mut data: Vec<Vec<f32>> = Vec::with_capacity(n_var);
                let mut mask: Vec<Vec<bool>> = Vec::with_capacity(n_var);

                for (vi, raw_ctx) in raw_variates.iter().enumerate() {
                    anyhow::ensure!(
                        raw_ctx.len() >= ctx_len,
                        "variate {vi} has fewer than {ctx_len} timesteps (needed to match variate 0)"
                    );
                    let start = raw_ctx.len() - ctx_len;
                    data.push(raw_ctx[start..].to_vec());
                    mask.push(vec![true; ctx_len]);
                }
                eprintln!("Running forecast ({ctx_len} context steps, {n_var} variate(s) → {horizon} future steps) …");
                let quantile_mat = model.forecast(&data, &mask, horizon).context("forecast")?;
                fc_outputs.push(zsfm_core::quantile_matrix_to_output(&quantile_mat, &quantile_levels, median_idx));
            }

            println!("{}", zsfm_core::forecast_response_json("toto", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
