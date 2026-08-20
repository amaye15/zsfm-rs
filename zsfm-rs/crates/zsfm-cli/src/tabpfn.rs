use std::path::PathBuf;

use anyhow::Context;
use clap::Subcommand;
use serde::Serialize;

use zsfm_tabpfn::config::TabPfnConfig;
use zsfm_tabpfn::TabPfnModel;

#[derive(Subcommand)]
pub enum Command {
    /// Print all tensor names in a local checkpoint file (safetensors, PyTorch
    /// pickle, GGUF, or npy/npz — format auto-detected).
    InspectTensors { path: PathBuf },
    /// Upload source + GGUF files to HuggingFace Hub.
    Upload {
        #[arg(short, long, default_value = "amaye15/tabpfn-gguf")]
        repo: String,
        #[arg(long, env = "HF_TOKEN")]
        token: String,
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Zero-shot classification from a GGUF file (classification only — regression's
    /// bar-distribution head is out of scope, see the crate's module docs).
    ///
    /// NON-COMMERCIAL WEIGHTS LICENSE (TabPFN-3 Non-Commercial License v1.0, Prior Labs GmbH):
    /// research, internal benchmarking, and evaluation only — no production, commercial, or
    /// hosted-service use without a separate commercial license from Prior Labs.
    ///
    /// Convert the checkpoint first with the generic converter (names pass through unchanged,
    /// so there's no `tabpfn convert` subcommand):
    ///   zsfm convert --repo Prior-Labs/tabpfn_3 --file classifier --format ckpt -o gguf/tabpfn-v3-f32.gguf
    ///
    /// Reads a JSON request from stdin:
    ///   {"x_support": [[...]], "y_support": [...], "x_query": [[...]], "n_classes": N}
    /// Outputs JSON: {"task":"classification","probabilities":[[...]]}.
    Infer {
        #[arg(short, long)]
        gguf: PathBuf,
    },
}

pub async fn run(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Upload { repo, token, root } => {
            zsfm_hub::upload_repo(&repo, &token, &root).await?;
        }

        Command::InspectTensors { path } => crate::common::inspect_tensors(&path)?,

        Command::Infer { gguf } => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf).context("read stdin")?;
            let req: serde_json::Value = serde_json::from_str(&buf).context("parse JSON input")?;

            let x_support = parse_matrix(&req["x_support"])?;
            let x_query = parse_matrix(&req["x_query"])?;
            let y_support: Vec<usize> = serde_json::from_value(req["y_support"].clone())
                .context("expected a 1D JSON integer array for `y_support`")?;
            let n_classes = req
                .get("n_classes")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| y_support.iter().copied().max().map(|m| m as u64 + 1).unwrap_or(1))
                as usize;

            let config = TabPfnConfig::v3_default();
            eprintln!("Loading model from {} …", gguf.display());
            let model = TabPfnModel::load(&gguf, config).context("load model")?;

            eprintln!(
                "Running classification ({} support / {} query rows, {n_classes} classes) …",
                x_support.len(),
                x_query.len()
            );
            let probabilities = model.predict_classification(&x_support, &y_support, &x_query, n_classes).context("predict")?;

            #[derive(Serialize)]
            struct Resp {
                task: &'static str,
                probabilities: Vec<Vec<f32>>,
            }
            println!("{}", serde_json::to_string_pretty(&Resp { task: "classification", probabilities })?);
        }
    }

    Ok(())
}

fn parse_matrix(val: &serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
    serde_json::from_value(val.clone()).context("expected a 2D JSON array")
}
