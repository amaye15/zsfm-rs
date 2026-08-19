mod cache;
mod data;
mod ensemble;
mod eval;
mod models;
mod report;
mod runner;

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rayon::prelude::*;

use models::{LoadedModel, ModelId};

#[derive(Parser)]
#[command(
    name = "zsfm-bench",
    about = "In-process rolling-window benchmark for the 11 zsfm time-series forecasters."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run one dataset/context/horizon config and print a results table.
    Run {
        #[arg(long, default_value = "ETTh1")]
        dataset: String,
        /// Comma-separated model names; default is all 11.
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        #[arg(long, default_value_t = 512)]
        context: usize,
        #[arg(long, default_value_t = 96)]
        horizon: usize,
        #[arg(long, default_value_t = 30)]
        windows: usize,
        #[arg(long)]
        ensemble_only: bool,
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
        #[arg(long, default_value = "../benchmark/data")]
        data_dir: PathBuf,
    },
    /// Run the full 21-dataset x horizon/context sweep and write benchmark.md.
    Report {
        #[arg(long, default_value_t = 30)]
        main_windows: usize,
        #[arg(long, default_value_t = 15)]
        sweep_windows: usize,
        #[arg(long)]
        no_cache: bool,
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
        #[arg(long, default_value = "models")]
        models_dir: PathBuf,
        #[arg(long, default_value = "../benchmark/data")]
        data_dir: PathBuf,
        #[arg(long, default_value = "../benchmark.md")]
        out: PathBuf,
        #[arg(long, default_value = "../benchmark/bench_cache.json")]
        cache: PathBuf,
        /// Cap rayon's global thread pool (default: all cores).
        #[arg(long)]
        jobs: Option<usize>,
    },
}

fn resolve_models(names: &Option<Vec<String>>) -> Result<Vec<ModelId>> {
    match names {
        None => Ok(ModelId::ALL.to_vec()),
        Some(names) => names
            .iter()
            .map(|n| ModelId::parse(n.trim()).with_context(|| format!("unknown model {n:?}")))
            .collect(),
    }
}

/// Load every requested model's canonical F32 GGUF exactly once, in
/// parallel. Models missing a cached GGUF are skipped with a warning
/// (rather than aborting the whole run) so a partial model set still works.
fn load_models(ids: &[ModelId], models_dir: &std::path::Path) -> HashMap<ModelId, LoadedModel> {
    ids.par_iter()
        .filter_map(|&id| {
            println!("Loading {id} …");
            match LoadedModel::load(id, models_dir) {
                Ok(m) => Some((id, m)),
                Err(e) => {
                    eprintln!("  skipping {id}: {e:#}");
                    None
                }
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            dataset,
            models,
            context,
            horizon,
            windows,
            ensemble_only,
            models_dir,
            data_dir,
        } => {
            let ids = resolve_models(&models)?;
            let loaded = load_models(&ids, &models_dir);
            let order: Vec<ModelId> = ids.into_iter().filter(|m| loaded.contains_key(m)).collect();
            anyhow::ensure!(
                !order.is_empty(),
                "no models loaded — run `zsfm <model> convert` first"
            );

            println!("\nDataset : {dataset}");
            println!("Context : {context}   Horizon : {horizon}   Windows : {windows}\n");

            let run = runner::run_config(
                &loaded, &order, &data_dir, &dataset, context, horizon, windows,
            )?;
            println!(
                "Series length: {}  |  Test rows: {}\n",
                run.series_len, run.test_rows
            );

            let col_w = 20;
            println!("{}", "=".repeat(68));
            println!(
                "{:<col_w$} {:>8} {:>8} {:>8}  {:>12}",
                "Model/Ensemble", "MAE", "RMSE", "MASE", "vs best solo"
            );
            println!("{}", "-".repeat(68));

            let best_solo_mae = order
                .iter()
                .filter_map(|m| run.solo.get(m.as_str()).map(|s| s.metrics.mae))
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);

            if !ensemble_only {
                println!("--- individual ---");
                for m in &order {
                    if let Some(s) = run.solo.get(m.as_str()) {
                        let delta = s.metrics.mae - best_solo_mae;
                        let lat = if s.lat_ms.is_finite() {
                            format!("  {:.0}ms", s.lat_ms)
                        } else {
                            String::new()
                        };
                        println!(
                            "{:<col_w$} {:>8.4} {:>8.4} {:>8.3}  {:>+.4}{lat}",
                            m.as_str(),
                            s.metrics.mae,
                            s.metrics.rmse,
                            s.metrics.mase,
                            delta
                        );
                    }
                }
                println!("--- ensembles ---");
            }
            for e in &run.ensembles {
                let delta = e.metrics.mae - best_solo_mae;
                println!(
                    "{:<col_w$} {:>8.4} {:>8.4} {:>8.3}  {:>+.4}",
                    e.name, e.metrics.mae, e.metrics.rmse, e.metrics.mase, delta
                );
            }
            println!("{}", "=".repeat(68));
            println!(
                "\nEnsemble key:  T=toto  C=chronos  F=timesfm  S=sundial  K=ttm  L=lag_llama  \
                 M=moment  O=moirai  P=moirai2  W=flowstate  X=tirex"
            );
            Ok(())
        }
        Command::Report {
            main_windows,
            sweep_windows,
            no_cache,
            models,
            models_dir,
            data_dir,
            out,
            cache,
            jobs,
        } => {
            if let Some(n) = jobs {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build_global()
                    .ok();
            }
            let ids = resolve_models(&models)?;
            let loaded = load_models(&ids, &models_dir);
            let order: Vec<ModelId> = ids.into_iter().filter(|m| loaded.contains_key(m)).collect();
            anyhow::ensure!(
                !order.is_empty(),
                "no models loaded — run `zsfm <model> convert` first"
            );
            if order.len() < ModelId::ALL.len() {
                println!(
                    "Note: only {}/{} models loaded ({}); report will be partial.",
                    order.len(),
                    ModelId::ALL.len(),
                    order
                        .iter()
                        .map(|m| m.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            let md = report::generate(
                &loaded,
                &order,
                &data_dir,
                &cache,
                main_windows,
                sweep_windows,
                no_cache,
            )?;
            std::fs::write(&out, &md).with_context(|| format!("write {}", out.display()))?;
            println!("\n✓ Wrote {}  ({} bytes)", out.display(), md.len());
            Ok(())
        }
    }
}
