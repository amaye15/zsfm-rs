use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::cache::{CachedRun, EnsembleResult, SoloResult};
use crate::data;
use crate::ensemble;
use crate::eval::{self, WindowResult};
use crate::models::{LoadedModel, ModelId};

fn combinations(items: &[ModelId], k: usize) -> Vec<Vec<ModelId>> {
    fn go(
        items: &[ModelId],
        k: usize,
        start: usize,
        cur: &mut Vec<ModelId>,
        out: &mut Vec<Vec<ModelId>>,
    ) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..items.len() {
            cur.push(items[i]);
            go(items, k, i + 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    go(items, k, 0, &mut Vec::new(), &mut out);
    out
}

/// Run every rolling window for every model, then every non-trivial ensemble
/// combo/strategy, for one (dataset, context, horizon, windows) config.
/// Models are evaluated in parallel (rayon) since each is fully independent
/// and already loaded — this is the "as fast as possible" core: no reload,
/// no subprocess, and the 11 models' inference work runs concurrently.
pub fn run_config(
    models: &HashMap<ModelId, LoadedModel>,
    model_order: &[ModelId],
    data_dir: &Path,
    dataset: &str,
    context: usize,
    horizon: usize,
    n_windows: usize,
) -> Result<CachedRun> {
    let spec = data::find(dataset).with_context(|| format!("unknown dataset {dataset}"))?;
    let series = data::load_series(data_dir, spec)?;

    let per_model: Vec<(ModelId, Vec<Option<WindowResult>>, SoloResult)> = model_order
        .par_iter()
        .filter_map(|&id| {
            let model = models.get(&id)?;
            let run = eval::run_model_windows(
                id.as_str(),
                model,
                &series,
                spec.test_rows,
                context,
                horizon,
                n_windows,
            );
            let valid: Vec<&WindowResult> = run.windows.iter().flatten().collect();
            let metrics = eval::metrics_from_windows(&valid);
            let solo = SoloResult {
                metrics,
                lat_ms: run.mean_lat_ms,
                errors: run.errors,
            };
            Some((id, run.windows, solo))
        })
        .collect();

    let mut all_windows: HashMap<ModelId, Vec<Option<WindowResult>>> = HashMap::new();
    let mut solo: HashMap<String, SoloResult> = HashMap::new();
    for (id, windows, s) in per_model {
        solo.insert(id.as_str().to_string(), s);
        all_windows.insert(id, windows);
    }

    let inv_mae: HashMap<ModelId, f64> = model_order
        .iter()
        .filter_map(|&id| {
            let m = solo.get(id.as_str())?;
            (m.metrics.mae.is_finite() && m.metrics.mae > 0.0).then_some((id, 1.0 / m.metrics.mae))
        })
        .collect();

    let mut ensembles = Vec::new();
    for size in 2..=model_order.len() {
        for combo in combinations(model_order, size) {
            for &(method, _abbrev) in ensemble::METHODS {
                if method == "trim" && size < 3 {
                    continue;
                }
                if ensemble::WEIGHT_METHODS.contains(&method) {
                    let n_weighted = combo.iter().filter(|m| inv_mae.contains_key(m)).count();
                    if n_weighted < 2 {
                        continue;
                    }
                }
                let ew = ensemble::ensemble_windows(&all_windows, &combo, method, Some(&inv_mae));
                if ew.is_empty() {
                    continue;
                }
                let refs: Vec<&WindowResult> = ew.iter().collect();
                let metrics = eval::metrics_from_windows(&refs);
                ensembles.push(EnsembleResult {
                    name: ensemble::label(&combo, method),
                    metrics,
                });
            }
        }
    }
    // `total_cmp` (not `partial_cmp().unwrap()`): a model or ensemble combination
    // can legitimately produce a NaN MAE (e.g. every window degenerate on a very
    // short dataset), and `partial_cmp` returns `None` for NaN, which would panic
    // `sort_by` and abort an entire multi-hour report run over one bad config.
    ensembles.sort_by(|a, b| a.metrics.mae.total_cmp(&b.metrics.mae));

    Ok(CachedRun {
        series_len: series.len(),
        test_rows: spec.test_rows,
        solo,
        ensembles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_count_matches_binomial() {
        let items = [
            ModelId::Toto,
            ModelId::Chronos,
            ModelId::TimesFM,
            ModelId::Sundial,
        ];
        assert_eq!(combinations(&items, 2).len(), 6); // C(4,2) = 6
        assert_eq!(combinations(&items, 4).len(), 1); // C(4,4) = 1
        assert_eq!(combinations(&items, 1).len(), 4); // C(4,1) = 4
    }

    #[test]
    fn combinations_are_order_preserving_subsets() {
        let items = [ModelId::Toto, ModelId::Chronos, ModelId::TimesFM];
        let combos = combinations(&items, 2);
        assert!(combos.contains(&vec![ModelId::Toto, ModelId::Chronos]));
        assert!(combos.contains(&vec![ModelId::Toto, ModelId::TimesFM]));
        assert!(combos.contains(&vec![ModelId::Chronos, ModelId::TimesFM]));
    }
}
