use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;

use crate::cache::{self, Cache, CachedRun};
use crate::data::DATASETS;
use crate::models::{LoadedModel, ModelId};
use crate::runner;

const HORIZONS: [usize; 4] = [96, 192, 336, 720];
const CONTEXTS: [usize; 4] = [96, 256, 512, 1024];
const MAIN_CONTEXT: usize = 512;
const MAIN_HORIZON: usize = 96;
const TOP_N_ENSEMBLES: usize = 10;

fn fmt(v: f64, decimals: usize) -> String {
    if v.is_nan() {
        "—".to_string()
    } else {
        format!("{v:.decimals$}")
    }
}

fn md_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str("| ");
    out.push_str(&headers.join(" | "));
    out.push_str(" |\n| ");
    out.push_str(
        &headers
            .iter()
            .enumerate()
            .map(|(i, _)| if i == 0 { ":---" } else { "---:" })
            .collect::<Vec<_>>()
            .join(" | "),
    );
    out.push_str(" |\n");
    for row in rows {
        out.push_str("| ");
        out.push_str(&row.join(" | "));
        out.push_str(" |\n");
    }
    out
}

fn bold_min(cells: &[String], raw: &[f64]) -> Vec<String> {
    let min_val = raw
        .iter()
        .cloned()
        .filter(|v| !v.is_nan())
        .fold(f64::INFINITY, f64::min);
    if !min_val.is_finite() {
        return cells.to_vec();
    }
    cells
        .iter()
        .zip(raw)
        .map(|(c, r)| {
            if (r - min_val).abs() < 1e-9 {
                format!("**{c}**")
            } else {
                c.clone()
            }
        })
        .collect()
}

fn ensure_ran(
    models: &HashMap<ModelId, LoadedModel>,
    model_order: &[ModelId],
    data_dir: &Path,
    cache: &mut Cache,
    configs: &[(String, usize, usize, usize)],
    phase_label: &str,
) {
    let todo: Vec<&(String, usize, usize, usize)> = configs
        .iter()
        .filter(|(ds, c, h, w)| cache.get(&cache::key(ds, *c, *h, *w)).is_none())
        .collect();
    if todo.is_empty() {
        return;
    }
    println!(
        "  {phase_label}: running {} configs (jobs run in parallel across models)…",
        todo.len()
    );
    let results: Vec<(String, CachedRun)> = todo
        .par_iter()
        .filter_map(|(ds, c, h, w)| {
            match runner::run_config(models, model_order, data_dir, ds, *c, *h, *w) {
                Ok(run) => {
                    println!("    [{ds}] ctx={c} h={h}  ({} models)", run.solo.len());
                    Some((cache::key(ds, *c, *h, *w), run))
                }
                Err(e) => {
                    eprintln!("    [{ds}] ctx={c} h={h}  ERROR — {e}");
                    None
                }
            }
        })
        .collect();
    for (k, run) in results {
        cache.insert(k, run);
    }
}

pub fn generate(
    models: &HashMap<ModelId, LoadedModel>,
    model_order: &[ModelId],
    data_dir: &Path,
    cache_path: &Path,
    main_windows: usize,
    sweep_windows: usize,
    no_cache: bool,
) -> Result<String> {
    let mut cache = if no_cache {
        Cache::default()
    } else {
        Cache::load(cache_path)
    };

    let ds_names: Vec<&str> = DATASETS.iter().map(|d| d.name).collect();

    println!("\n=== PHASE 1: main benchmark (ctx={MAIN_CONTEXT}, h={MAIN_HORIZON}) ===");
    let phase1: Vec<_> = ds_names
        .iter()
        .map(|d| (d.to_string(), MAIN_CONTEXT, MAIN_HORIZON, main_windows))
        .collect();
    ensure_ran(
        models,
        model_order,
        data_dir,
        &mut cache,
        &phase1,
        "PHASE 1",
    );
    cache.save(cache_path)?;

    println!("\n=== PHASE 2: horizon sweep (ctx={MAIN_CONTEXT}, h={HORIZONS:?}) ===");
    let phase2: Vec<_> = ds_names
        .iter()
        .flat_map(|d| {
            HORIZONS
                .iter()
                .map(move |&h| (d.to_string(), MAIN_CONTEXT, h, sweep_windows))
        })
        .collect();
    ensure_ran(
        models,
        model_order,
        data_dir,
        &mut cache,
        &phase2,
        "PHASE 2",
    );
    cache.save(cache_path)?;

    println!("\n=== PHASE 3: context sweep (h={MAIN_HORIZON}, ctx={CONTEXTS:?}) ===");
    let phase3: Vec<_> = ds_names
        .iter()
        .flat_map(|d| {
            CONTEXTS
                .iter()
                .map(move |&c| (d.to_string(), c, MAIN_HORIZON, sweep_windows))
        })
        .collect();
    ensure_ran(
        models,
        model_order,
        data_dir,
        &mut cache,
        &phase3,
        "PHASE 3",
    );
    cache.save(cache_path)?;

    let get = |ds: &str, c: usize, h: usize, w: usize| -> Option<&CachedRun> {
        cache.get(&cache::key(ds, c, h, w))
    };

    let mut md = String::new();
    let today = std::time::SystemTime::now();
    let today_str = humantime_date(today);

    md.push_str("# Time Series Forecasting Benchmark\n\n");
    md.push_str(&format!("**Date:** {today_str}  \n"));
    md.push_str(&format!(
        "**Main evaluation:** context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, {main_windows} rolling windows  \n"
    ));
    md.push_str(&format!(
        "**Sweep evaluations:** {sweep_windows} rolling windows  \n"
    ));
    md.push_str(
        "**Metric:** MAE (lower is better); MASE for cross-dataset comparison (scale-free)\n\n",
    );
    md.push_str("**Models:**\n\n");
    md.push_str(&md_table(
        &["Model", "Architecture"],
        &model_order
            .iter()
            .map(|m| vec![m.label().to_string(), model_arch(*m).to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push('\n');
    md.push_str("**Ensemble strategies (all non-trivial subsets, size ≥ 2):**\n\n");
    md.push_str(&md_table(
        &["Symbol", "Strategy"],
        &crate::ensemble::METHODS
            .iter()
            .map(|(m, a)| vec![format!("`({a})`"), strategy_desc(m).to_string()])
            .collect::<Vec<_>>(),
    ));
    md.push_str("\n---\n\n## Table of Contents\n\n");
    md.push_str(
        "1. [Individual Models — MAE](#individual-models--mae)\n\
         2. [Individual Models — MASE](#individual-models--mase-scale-free)\n\
         3. [Inference Latency](#inference-latency)\n\
         4. [Best Ensemble vs Best Solo](#best-ensemble-vs-best-solo)\n\
         5. [Horizon Scaling](#horizon-scaling)\n\
         6. [Context Length Scaling](#context-length-scaling)\n\
         7. [Per-Dataset Detail](#per-dataset-detail)\n\
         8. [Methodology](#methodology)\n\n---\n\n",
    );

    // Section 1: MAE
    md.push_str("## Individual Models — MAE\n\n");
    md.push_str(&format!(
        "> context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, {main_windows} windows. **Bold** = best on that dataset.\n\n"
    ));
    let mut rows = Vec::new();
    for spec in DATASETS {
        let Some(run) = get(spec.name, MAIN_CONTEXT, MAIN_HORIZON, main_windows) else {
            continue;
        };
        let raw: Vec<f64> = model_order
            .iter()
            .map(|m| {
                run.solo
                    .get(m.as_str())
                    .map(|s| s.metrics.mae)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        let cells = bold_min(&raw.iter().map(|v| fmt(*v, 4)).collect::<Vec<_>>(), &raw);
        let winner = model_order
            .iter()
            .zip(&raw)
            .filter(|(_, v)| !v.is_nan())
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(m, _)| m.as_str())
            .unwrap_or("—");
        let mut row = vec![
            spec.name.to_string(),
            spec.freq.to_string(),
            spec.domain.to_string(),
        ];
        row.extend(cells);
        row.push(winner.to_string());
        rows.push(row);
    }
    let mut headers = vec!["Dataset", "Freq", "Domain"];
    let model_labels: Vec<&str> = model_order.iter().map(|m| m.label()).collect();
    headers.extend(model_labels.iter());
    headers.push("Winner");
    md.push_str(&md_table(&headers, &rows));
    md.push('\n');

    // Section 2: MASE
    md.push_str("## Individual Models — MASE (scale-free)\n\n");
    md.push_str(
        "> MASE < 1 means the model beats a naïve 1-step random walk. **Bold** = best.\n\n",
    );
    let mut rows = Vec::new();
    for spec in DATASETS {
        let Some(run) = get(spec.name, MAIN_CONTEXT, MAIN_HORIZON, main_windows) else {
            continue;
        };
        let raw: Vec<f64> = model_order
            .iter()
            .map(|m| {
                run.solo
                    .get(m.as_str())
                    .map(|s| s.metrics.mase)
                    .unwrap_or(f64::NAN)
            })
            .collect();
        let cells = bold_min(&raw.iter().map(|v| fmt(*v, 3)).collect::<Vec<_>>(), &raw);
        let winner = model_order
            .iter()
            .zip(&raw)
            .filter(|(_, v)| !v.is_nan())
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(m, _)| m.as_str())
            .unwrap_or("—");
        let mut row = vec![spec.name.to_string()];
        row.extend(cells);
        row.push(winner.to_string());
        rows.push(row);
    }
    let mut headers = vec!["Dataset"];
    headers.extend(model_labels.iter());
    headers.push("Winner");
    md.push_str(&md_table(&headers, &rows));
    md.push('\n');

    // Section 3: latency
    md.push_str("## Inference Latency\n\n");
    md.push_str(&format!(
        "> Milliseconds per window (context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}).\n\n"
    ));
    let mut rows = Vec::new();
    for spec in DATASETS {
        let Some(run) = get(spec.name, MAIN_CONTEXT, MAIN_HORIZON, main_windows) else {
            continue;
        };
        let mut row = vec![spec.name.to_string()];
        for m in model_order {
            let cell = run
                .solo
                .get(m.as_str())
                .and_then(|s| s.lat_ms.is_finite().then_some(s.lat_ms))
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "—".to_string());
            row.push(cell);
        }
        rows.push(row);
    }
    let mut headers = vec!["Dataset"];
    headers.extend(model_labels.iter());
    md.push_str(&md_table(&headers, &rows));
    md.push('\n');

    // Section 4: best ensemble vs best solo
    md.push_str("## Best Ensemble vs Best Solo\n\n");
    md.push_str("> Δ MAE = ensemble MAE − best solo MAE. Negative = ensemble wins.\n\n");
    let mut rows = Vec::new();
    for spec in DATASETS {
        let Some(run) = get(spec.name, MAIN_CONTEXT, MAIN_HORIZON, main_windows) else {
            continue;
        };
        if run.ensembles.is_empty() {
            continue;
        }
        let best_solo = model_order
            .iter()
            .filter_map(|m| {
                run.solo
                    .get(m.as_str())
                    .map(|s| (m.as_str(), s.metrics.mae))
            })
            .filter(|(_, mae)| mae.is_finite())
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((solo_name, solo_mae)) = best_solo else {
            continue;
        };
        let best_ens = &run.ensembles[0];
        let delta = best_ens.metrics.mae - solo_mae;
        let pct = (delta.abs() / solo_mae) * 100.0;
        let direction = if delta < 0.0 {
            format!("↓ {pct:.1}%")
        } else {
            format!("↑ {pct:.1}%")
        };
        rows.push(vec![
            spec.name.to_string(),
            solo_name.to_string(),
            fmt(solo_mae, 4),
            best_ens.name.clone(),
            fmt(best_ens.metrics.mae, 4),
            format!("{delta:+.4}"),
            direction,
        ]);
    }
    md.push_str(&md_table(
        &[
            "Dataset",
            "Best Solo",
            "Solo MAE",
            "Best Ensemble",
            "Ensemble MAE",
            "Δ MAE",
            "Change",
        ],
        &rows,
    ));
    md.push_str("\n---\n\n");

    // Section 5: horizon scaling
    md.push_str("## Horizon Scaling\n\n");
    md.push_str(&format!(
        "> context={MAIN_CONTEXT} fixed, horizons={HORIZONS:?}, {sweep_windows} windows. **Bold** = best horizon for that dataset/model.\n\n"
    ));
    md.push_str("### Average MASE across all datasets by horizon\n\n");
    md.push_str(&avg_sweep_table(
        &cache,
        model_order,
        "h",
        &HORIZONS,
        sweep_windows,
    ));
    md.push('\n');
    md.push_str("---\n\n");

    // Section 6: context scaling
    md.push_str("## Context Length Scaling\n\n");
    md.push_str(&format!(
        "> horizon={MAIN_HORIZON} fixed, contexts={CONTEXTS:?}, {sweep_windows} windows. **Bold** = best context for that dataset/model.\n\n"
    ));
    md.push_str("### Average MASE across all datasets by context length\n\n");
    md.push_str(&avg_sweep_table(
        &cache,
        model_order,
        "ctx",
        &CONTEXTS,
        sweep_windows,
    ));
    md.push('\n');
    md.push_str("---\n\n");

    // Section 7: per-dataset detail
    md.push_str("## Per-Dataset Detail\n\n");
    md.push_str(&format!(
        "> context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, {main_windows} windows.\n\n"
    ));
    for spec in DATASETS {
        let Some(run) = get(spec.name, MAIN_CONTEXT, MAIN_HORIZON, main_windows) else {
            continue;
        };
        md.push_str(&format!("### {}\n\n", spec.name));
        md.push_str(&format!(
            "**Frequency:** {}  |  **Domain:** {}  |  **Series length:** {}  |  **Test rows:** {}\n\n",
            spec.freq, spec.domain, run.series_len, run.test_rows
        ));
        md.push_str("**Individual models:**\n\n");
        let best = model_order
            .iter()
            .filter_map(|m| run.solo.get(m.as_str()).map(|s| s.metrics.mae))
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let mut solo_rows = Vec::new();
        for m in model_order {
            let Some(s) = run.solo.get(m.as_str()) else {
                continue;
            };
            let mut mae = fmt(s.metrics.mae, 4);
            if (s.metrics.mae - best).abs() < 1e-9 {
                mae = format!("**{mae}**");
            }
            let lat = if s.lat_ms.is_finite() {
                format!("{:.0}ms", s.lat_ms)
            } else {
                "—".to_string()
            };
            solo_rows.push(vec![
                m.as_str().to_string(),
                mae,
                fmt(s.metrics.rmse, 4),
                fmt(s.metrics.mase, 3),
                lat,
            ]);
        }
        md.push_str(&md_table(
            &["Model", "MAE", "RMSE", "MASE", "Latency"],
            &solo_rows,
        ));
        md.push('\n');

        if !run.ensembles.is_empty() {
            md.push_str(&format!("**Top {TOP_N_ENSEMBLES} ensembles:**\n\n"));
            let ens_rows: Vec<Vec<String>> = run
                .ensembles
                .iter()
                .take(TOP_N_ENSEMBLES)
                .map(|e| {
                    vec![
                        e.name.clone(),
                        fmt(e.metrics.mae, 4),
                        fmt(e.metrics.rmse, 4),
                        fmt(e.metrics.mase, 3),
                        format!("{:+.4}", e.metrics.mae - best),
                    ]
                })
                .collect();
            md.push_str(&md_table(
                &["Ensemble", "MAE", "RMSE", "MASE", "Δ vs best solo"],
                &ens_rows,
            ));
            md.push('\n');
        }
    }

    // Section 8: methodology
    md.push_str("---\n\n## Methodology\n\n");
    md.push_str(
        "- All models run in-process via `zsfm-bench`, linking the `zsfm-rs` model crates directly \
         through the shared `zsfm_core::Forecaster` load-once interface — no subprocess, no per-call \
         GGUF reload. F32 GGUF weights (the canonical cache `zsfm <model> convert` produces).\n\
         - Rolling windows evenly spaced over the held-out test split of each dataset.\n\
         - **MAE** = mean absolute error averaged across all windows.\n\
         - **RMSE** = root mean squared error averaged across windows.\n\
         - **MASE** = MAE / mean-absolute-1-step-diff of the context window (scale-free).\n\
         - Ensemble weights (`w`, `sfx`) derived from overall MAE on the same evaluation set (in-sample). \
         `onl`/`ada`/`smo`/`sel`/`wh`/`uq` weights update window-by-window, causally.\n",
    );
    md.push_str(&format!("- **Horizon sweep:** context={MAIN_CONTEXT} fixed, horizons={HORIZONS:?}, {sweep_windows} windows.\n"));
    md.push_str(&format!("- **Context sweep:** horizon={MAIN_HORIZON} fixed, contexts={CONTEXTS:?}, {sweep_windows} windows.\n"));
    md.push_str("- Datasets: ETDataset (ETTh/m), Monash (electricity, weather), Informer bundle (ILI, Exchange, Traffic), Jena Climate.\n");

    Ok(md)
}

fn avg_sweep_table(
    cache: &Cache,
    model_order: &[ModelId],
    axis: &str,
    values: &[usize],
    windows: usize,
) -> String {
    let mut rows = Vec::new();
    for m in model_order {
        let mut raw = Vec::new();
        for &v in values {
            let mut vals = Vec::new();
            for spec in DATASETS {
                let (c, h) = if axis == "h" {
                    (MAIN_CONTEXT, v)
                } else {
                    (v, MAIN_HORIZON)
                };
                if let Some(run) = cache.get(&cache::key(spec.name, c, h, windows)) {
                    if let Some(s) = run.solo.get(m.as_str()) {
                        if s.metrics.mase.is_finite() {
                            vals.push(s.metrics.mase);
                        }
                    }
                }
            }
            raw.push(if vals.is_empty() {
                f64::NAN
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            });
        }
        let cells = bold_min(&raw.iter().map(|v| fmt(*v, 3)).collect::<Vec<_>>(), &raw);
        let mut row = vec![m.label().to_string()];
        row.extend(cells);
        rows.push(row);
    }
    let mut headers = vec!["Model".to_string()];
    headers.extend(values.iter().map(|v| format!("{axis}={v}")));
    let header_refs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    md_table(&header_refs, &rows)
}

fn model_arch(m: ModelId) -> &'static str {
    match m {
        ModelId::Toto => "Quantile transformer (Datadog)",
        ModelId::Chronos => "Tokenised probabilistic transformer (Amazon)",
        ModelId::TimesFM => "Patch-based decoder (Google)",
        ModelId::Sundial => "Flow-matching decoder transformer (thuml)",
        ModelId::Ttm => "Tiny time-mixer (IBM)",
        ModelId::LagLlama => "Lag-feature LLM decoder",
        ModelId::Moment => "Masked patch encoder, T5 attention (CMU)",
        ModelId::Moirai => "Universal forecasting encoder (Salesforce)",
        ModelId::Moirai2 => "Universal forecasting decoder (Salesforce)",
        ModelId::FlowState => "SSM encoder-decoder with Legendre quantile basis (IBM)",
        ModelId::Tirex => "Patch-based encoder with RoPE (NX-AI)",
    }
}

fn strategy_desc(method: &str) -> &'static str {
    match method {
        "mean" => "Simple mean",
        "median" => "Per-timestep median",
        "weighted" => "Inverse-MAE weighted",
        "trim" => "Trimmed mean (drop min+max per timestep)",
        "softmax" => "Softmax-weighted (sharpened inverse-MAE)",
        "geometric" => "Sign-preserving geometric mean",
        "online" => "Online adaptive (weights from previous window's error)",
        "adaptive" => "Adaptive Hedge — multiplicative weight update, optimal η=√(2 ln N/T)",
        "smooth" => "Smooth online — EMA of inverse-MAE (blend of online and Hedge)",
        "select" => "Model selection — greedy pick of single best model by EMA-MAE",
        "per_horiz" => "Per-horizon weights — distinct model mix per forecast step",
        "uncertain" => "Uncertainty-weighted — IQR for quantile models, MAE-fallback for others",
        _ => "",
    }
}

fn humantime_date(t: std::time::SystemTime) -> String {
    let secs = t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let days = secs / 86400;
    // Simple civil-from-days (Howard Hinnant's algorithm), UTC, no external dep.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}
