use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_chronos::config::Chronos2Config;
use zsfm_chronos::convert::{convert, ConvertOptions};
use zsfm_chronos::infer::ChronosModel;
use zsfm_gguf::GGMLType;

#[derive(Subcommand)]
pub enum Command {
    /// Download a Chronos-2 model from HuggingFace and convert it to GGUF.
    Convert {
        #[arg(short, long, default_value = "amazon/chronos-2")]
        model: String,
        #[arg(short, long, default_value = "gguf/chronos-f16.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f16")]
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
    /// Run Chronos-2 forecasting from a GGUF file.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with all quantile levels.
    /// The `point` field contains the median quantile (q0.5). Univariate only.
    Infer {
        #[arg(short, long, default_value = "gguf/chronos-f16.gguf")]
        gguf: PathBuf,
        #[arg(long, default_value = "models/config.json")]
        config: PathBuf,
    },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/chronos-rs-gguf")]
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
            DtypeArg::Q8 => GGMLType::Q8_0,
        }
    }
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token } => {
            println!("Downloading {model} into {} …", model_dir.display());
            let files = zsfm_hub::download_model(&model, token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config_str = std::fs::read_to_string(&files.config_json).context("read config.json")?;
            let config = Chronos2Config::from_json(&config_str).context("parse config.json")?;

            println!(
                "Config: {} layers, d_model={}, d_ff={}, {} heads, patch_size={}",
                config.num_layers,
                config.d_model,
                config.d_ff,
                config.num_heads,
                config.chronos_config.input_patch_size,
            );

            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&model, &files, &config, &opts, &output)?;
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

        Command::Infer { gguf, config } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config).with_context(|| format!("read {}", config.display()))?;
            let c2_config = Chronos2Config::from_json(&config_str).context("parse config.json")?;

            eprintln!("Loading model from {} …", gguf.display());
            let model = ChronosModel::builder(&gguf).config_from(&c2_config).build().context("load model")?;

            let patch_size = model.config.patch_size();
            let patch_stride = model.config.patch_stride();
            let max_ctx = model.config.context_length();
            let quantile_levels = model.config.quantiles().to_vec();
            let median_idx = quantile_levels
                .iter()
                .position(|&q| (q - 0.5).abs() < 1e-6)
                .unwrap_or(quantile_levels.len() / 2);

            let mut fc_outputs = Vec::new();
            let mut total_ctx: usize = 0;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "Chronos-2 only supports univariate forecasting (1 variate per context)"
                );
                let mut ctx = raw_variates[0].clone();
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                total_ctx += ctx.len();

                // Trim to context window aligned to patch_stride
                let total_len = ctx.len();
                let usable = total_len.min(max_ctx);
                let ctx_len = (usable / patch_stride) * patch_stride;
                anyhow::ensure!(
                    ctx_len >= patch_size,
                    "Need at least {patch_size} timesteps of context, got {ctx_len}."
                );
                let start = total_len.saturating_sub(ctx_len);
                ctx = ctx[start..].to_vec();

                eprintln!("Running forecast ({} context steps → {horizon} future steps) …", ctx.len());
                let quantile_mat = model.forecast(&ctx, horizon).context("forecast")?;
                // [n_q][horizon] -> [n_q][1 variate][horizon] for the shared envelope helper.
                let qmat: zsfm_core::QuantileMatrix = quantile_mat.into_iter().map(|row| vec![row]).collect();
                fc_outputs.push(zsfm_core::quantile_matrix_to_output(&qmat, &quantile_levels, median_idx));
            }

            println!("{}", zsfm_core::forecast_response_json("chronos", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
