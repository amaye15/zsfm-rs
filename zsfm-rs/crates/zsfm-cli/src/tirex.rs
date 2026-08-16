use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Subcommand, ValueEnum};

use zsfm_gguf::GGMLType;
use zsfm_tirex::config::TiRexConfig;
use zsfm_tirex::convert::{convert, ConvertOptions};
use zsfm_tirex::infer::TiRexModel;

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
        #[arg(short, long, default_value = "amaye15/tirex-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        /// Directory tree to upload. Defaults to the current directory.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Download TiRex from HuggingFace and convert to GGUF.
    ///
    /// Downloads model.ckpt and reads it directly via candle's pickle reader —
    /// no Python or intermediate extraction step required.
    Convert {
        #[arg(short, long, default_value = "NX-AI/TiRex")]
        model: String,
        #[arg(short, long, default_value = "gguf/tirex-f32.gguf")]
        output: PathBuf,
        #[arg(long, default_value = "f32")]
        dtype: DtypeArg,
        #[arg(long, default_value = "models")]
        model_dir: PathBuf,
        #[arg(long, env = "HF_TOKEN")]
        token: Option<String>,
    },
    /// Run TiRex forecasting from a GGUF file. Univariate only.
    ///
    /// Reads a JSON request from stdin: {"context": [...], "horizon": N}
    /// Outputs a JSON forecast in OpenAI-compatible format with all quantile levels.
    /// The `point` field contains the median quantile (q0.5).
    Infer {
        #[arg(short, long, default_value = "gguf/tirex-f32.gguf")]
        gguf: PathBuf,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::Convert { model, output, dtype, model_dir, token } => {
            println!("Downloading {model} into {} …", model_dir.display());
            let ckpt_path = zsfm_hub::download_file(&model, "model.ckpt", token.as_deref(), &model_dir)
                .await
                .context("download failed")?;

            let config = TiRexConfig::default_from_ckpt();
            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&ckpt_path, &config, &opts, &output)?;
        }

        Command::Infer { gguf } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;
            let contexts = zsfm_core::parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;
            let config = TiRexConfig::default_from_ckpt();

            eprintln!("Loading model from {} …", gguf.display());
            let model = TiRexModel::load(&gguf, config.clone()).context("load model")?;

            let mut fc_outputs = Vec::new();
            let mut total_ctx = 0usize;
            for raw_variates in &contexts {
                anyhow::ensure!(
                    raw_variates.len() == 1,
                    "TiRex only supports univariate forecasting (1 variate per context)"
                );
                let ctx = &raw_variates[0];
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                total_ctx += ctx.len();
                let (quantiles, mean) = model.forecast(ctx, horizon).context("forecast")?;
                let mut q_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
                for (i, q) in config.quantiles.iter().enumerate() {
                    q_map.insert(format!("{q:.2}"), quantiles[i].clone());
                }
                fc_outputs.push(zsfm_core::ForecastOutput::Univariate { point: mean, quantiles: q_map });
            }
            println!("{}", zsfm_core::forecast_response_json("tirex", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}
