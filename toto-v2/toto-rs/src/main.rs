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

use config::TotoConfig;
use convert::{convert, ConvertOptions};
use download::download_model;
use gguf::GGMLType;
use infer::{InferConfig, TotoModel};

#[derive(Parser)]
#[command(name = "toto-rs", about = "Convert and run Datadog/Toto-2.0-2.5B")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        /// HuggingFace repo to upload to (owner/name).
        #[arg(short, long, default_value = "amaye15/toto-gguf")]
        repo: String,
        /// HuggingFace API token (or set HF_TOKEN env var).
        #[arg(long, env = "HF_TOKEN")]
        token: String,
    },
    /// Download a Toto model from HuggingFace and convert it to GGUF.
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
    },
    /// Print all tensor names in a local safetensors file.
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
    ///
    /// Univariate output has "point"/"quantiles" in each choice.
    /// Multivariate output has "variates": [{"point":..., "quantiles":...}, ...].
    Infer {
        /// Path to the GGUF file.
        #[arg(short, long, default_value = "gguf/toto-2.5b-f16.gguf")]
        gguf: PathBuf,

        /// Path to config.json from the original HuggingFace model.
        #[arg(long, default_value = "models/config.json")]
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
}

#[derive(Clone, ValueEnum)]
enum DtypeArg {
    F32,
    F16,
    Bf16,
    Q8,
}

impl From<DtypeArg> for GGMLType {
    fn from(d: DtypeArg) -> Self {
        match d {
            DtypeArg::F32 => GGMLType::F32,
            DtypeArg::F16 => GGMLType::F16,
            DtypeArg::Bf16 => GGMLType::BF16,
            DtypeArg::Q8 => GGMLType::Q8_0,
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
            let config = TotoConfig::from_json(&config_str)
                .context("parse config.json")?;

            println!(
                "Config: {} layers, hidden={}, heads={}",
                config.num_hidden_layers, config.hidden_size, config.num_attention_heads
            );

            let opts = ConvertOptions { output_dtype: dtype.into() };
            convert(&model, &files, &config, &opts, &output)?;
            println!("Wrote {}", output.display());
        }

        Command::InspectTensors { path } => {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let tensors = safetensors::SafeTensors::deserialize(&bytes)
                .context("deserialize safetensors")?;
            println!("Tensors in {}:", path.display());
            let mut names: Vec<_> = tensors.names().into_iter().collect();
            names.sort();
            for name in names {
                let t = tensors.tensor(name).unwrap();
                println!("  {name:60} {:?} {:?}", t.dtype(), t.shape());
            }
        }

        Command::Infer { gguf, config, context_length, use_f64 } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            // contexts: [batch][variate][time] — univariate is [batch][1][time]
            let contexts = parse_mv_contexts(req["context"].clone())?;
            let horizon: usize = req["horizon"].as_u64().context("horizon must be a positive integer")? as usize;

            let config_str = std::fs::read_to_string(&config)
                .with_context(|| format!("read {}", config.display()))?;
            let toto_config: serde_json::Value = serde_json::from_str(&config_str)?;

            let infer_config = InferConfig {
                d_model: toto_config["d_model"].as_u64().unwrap_or(2048) as usize,
                num_layers: toto_config["num_layers"].as_u64().unwrap_or(48) as usize,
                num_heads: toto_config["num_heads"].as_u64().unwrap_or(32) as usize,
                num_groups: toto_config["num_groups"].as_u64().unwrap_or(32) as usize,
                qk_dim: toto_config["qk_dim"].as_u64().unwrap_or(64) as usize,
                v_dim: toto_config["v_dim"].as_u64().unwrap_or(64) as usize,
                patch_size: toto_config["patch_size"].as_u64().unwrap_or(32) as usize,
                norm_eps: toto_config["norm_eps"].as_f64().unwrap_or(5e-4),
                layer_group_size: toto_config["layer_group_size"].as_u64().unwrap_or(48) as usize,
                num_variate_layers_per_group: toto_config["num_variate_layers_per_group"]
                    .as_u64().unwrap_or(1) as usize,
                variate_layer_first: toto_config["variate_layer_first"].as_bool().unwrap_or(false),
                use_xpos: toto_config["use_xpos"].as_bool().unwrap_or(true),
                residual_mult: toto_config["residual_mult"].as_f64().unwrap_or(0.75),
                residual_attn_ratio: toto_config["residual_attn_ratio"].as_f64().unwrap_or(5.136215466577748),
                compute_f64: use_f64,
            };

            let patch_size = infer_config.patch_size;
            let max_ctx = context_length.unwrap_or(4096);
            let quantile_levels = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
            let median_idx = 4; // q0.5 is at index 4

            eprintln!("Loading model from {} …", gguf.display());
            let model = TotoModel::load(&gguf, infer_config)
                .context("load model")?;

            let mut fc_outputs: Vec<ForecastOutput> = Vec::new();
            let mut total_ctx: usize = 0;

            for raw_variates in &contexts {
                let n_var = raw_variates.len();
                anyhow::ensure!(n_var > 0, "each context must have at least one variate");

                // Determine patch-aligned ctx_len from the first variate, apply to all
                let v0_len = raw_variates[0].len();
                let ctx_len = (v0_len.min(max_ctx) / patch_size) * patch_size;
                anyhow::ensure!(
                    ctx_len > 0,
                    "context too short — need at least {patch_size} timesteps"
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
                total_ctx += ctx_len * n_var;

                eprintln!("Running forecast ({ctx_len} context steps, {n_var} variate(s) → {horizon} future steps) …");
                // quantile_mat: [9_quantiles][n_var][prediction_length]
                let quantile_mat = model.forecast(&data, &mask, horizon).context("forecast")?;

                if n_var == 1 {
                    // Univariate: standard point/quantiles format (backward-compatible)
                    let point = quantile_mat.get(median_idx)
                        .and_then(|v| v.first())
                        .cloned()
                        .unwrap_or_default();
                    let mut quantiles = BTreeMap::new();
                    for (qi, &level) in quantile_levels.iter().enumerate() {
                        if let Some(var_row) = quantile_mat.get(qi) {
                            if let Some(q) = var_row.first() {
                                quantiles.insert(format!("{level:.2}"), q.clone());
                            }
                        }
                    }
                    fc_outputs.push(ForecastOutput::Univariate { point, quantiles });
                } else {
                    // Multivariate: variates array, one entry per variate
                    let mut var_forecasts: Vec<VariateForecast> = Vec::with_capacity(n_var);
                    for vi in 0..n_var {
                        let point = quantile_mat.get(median_idx)
                            .and_then(|v| v.get(vi))
                            .cloned()
                            .unwrap_or_default();
                        let mut quantiles = BTreeMap::new();
                        for (qi, &level) in quantile_levels.iter().enumerate() {
                            if let Some(var_row) = quantile_mat.get(qi) {
                                if let Some(q) = var_row.get(vi) {
                                    quantiles.insert(format!("{level:.2}"), q.clone());
                                }
                            }
                        }
                        var_forecasts.push(VariateForecast { point, quantiles });
                    }
                    fc_outputs.push(ForecastOutput::Multivariate { variates: var_forecasts });
                }
            }

            println!("{}", forecast_json("toto", total_ctx, horizon, fc_outputs)?);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(untagged)]
enum ForecastOutput {
    Univariate {
        point: Vec<f32>,
        quantiles: BTreeMap<String, Vec<f32>>,
    },
    Multivariate {
        variates: Vec<VariateForecast>,
    },
}

#[derive(Serialize)]
struct VariateForecast {
    point: Vec<f32>,
    quantiles: BTreeMap<String, Vec<f32>>,
}

// ---------------------------------------------------------------------------
// JSON input parsing
// ---------------------------------------------------------------------------

/// Parse the `context` field into [batch][variate][time].
///
/// Accepted shapes:
///   [t0, t1, ...]                    → batch=1, n_var=1 (flat single series)
///   [[t0, t1, ...], ...]             → batch=N, n_var=1 (batch of univariate)
///   [[[v0_t0, ...], [v1_t0, ...]], ...] → batch=N, n_var=M (batch of multivariate)
fn parse_mv_contexts(val: serde_json::Value) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    match val {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            anyhow::bail!("context must be a non-empty array")
        }
        serde_json::Value::Array(arr) => {
            let first = arr.first().unwrap();
            if !first.is_array() {
                // Flat: [t0, t1, ...] → batch=1, n_var=1
                let ctx = serde_json::from_value::<Vec<f32>>(serde_json::Value::Array(arr))
                    .context("context must be a JSON array of numbers")?;
                Ok(vec![vec![ctx]])
            } else if first
                .as_array()
                .and_then(|a| a.first())
                .map(|v| v.is_array())
                .unwrap_or(false)
            {
                // 3D: [batch][variate][time]
                arr.into_iter()
                    .enumerate()
                    .map(|(i, batch_item)| {
                        serde_json::from_value::<Vec<Vec<f32>>>(batch_item)
                            .with_context(|| format!("context[{i}] must be an array of variate arrays"))
                    })
                    .collect()
            } else {
                // 2D: [batch][time] → each series is n_var=1
                arr.into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let series = serde_json::from_value::<Vec<f32>>(v)
                            .with_context(|| format!("context[{i}] must be an array of numbers"))?;
                        Ok(vec![series])
                    })
                    .collect()
            }
        }
        _ => anyhow::bail!("context must be a JSON array"),
    }
}

// ---------------------------------------------------------------------------
// JSON response serialisation
// ---------------------------------------------------------------------------

fn forecast_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    fc_outputs: Vec<ForecastOutput>,
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
        choices: fc_outputs.into_iter().enumerate().map(|(i, forecast)| Choice {
            index: i,
            forecast,
            finish_reason: "stop",
        }).collect(),
        usage: Usage { context_length, forecast_length },
    };

    Ok(serde_json::to_string_pretty(&resp)?)
}
