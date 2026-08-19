use std::time::Instant;

use crate::models::LoadedModel;

/// One rolling window's result: model's point forecast, ground truth, and the
/// context it was conditioned on (context is kept because MASE's naive-error
/// denominator is computed from it).
#[derive(Clone)]
pub struct WindowResult {
    pub point: Vec<f32>,
    pub truth: Vec<f32>,
    pub context: Vec<f32>,
    /// Mean q0.90-q0.10 width, NAN if this model doesn't produce quantiles.
    pub iqr: f32,
}

pub struct ModelRunResult {
    /// Length `n_windows`; `None` at positions that were skipped (too little
    /// context) or failed (model returned `Err`, or a too-short forecast).
    pub windows: Vec<Option<WindowResult>>,
    pub mean_lat_ms: f64,
    pub errors: usize,
}

/// numpy's `np.linspace(start, stop, num, dtype=int)`: `num` evenly spaced
/// points from `start` to `stop` inclusive, then truncated to an integer.
fn linspace_int(start: i64, stop: i64, num: usize) -> Vec<usize> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start.max(0) as usize];
    }
    // When stop == start, step is 0.0 and every element correctly repeats `start`
    // (numpy's `linspace(3, 3, 4)` == `[3, 3, 3, 3]`, not a single collapsed value).
    let step = (stop - start) as f64 / (num - 1) as f64;
    (0..num)
        .map(|i| {
            let v = start as f64 + i as f64 * step;
            v.max(0.0) as usize
        })
        .collect()
}

/// Run every rolling window for one (model, dataset, context, horizon)
/// configuration. `model` is loaded once by the caller and reused across
/// every dataset/config combination — no per-call reload, no subprocess.
pub fn run_model_windows(
    model_name: &str,
    model: &LoadedModel,
    series: &[f32],
    test_rows: usize,
    context_len: usize,
    horizon: usize,
    n_windows: usize,
) -> ModelRunResult {
    let train_end = series.len().saturating_sub(test_rows);
    let max_start = series.len().saturating_sub(horizon);

    let mut windows = vec![None; n_windows];
    if max_start == 0 || max_start <= train_end {
        return ModelRunResult {
            windows,
            mean_lat_ms: f64::NAN,
            errors: n_windows,
        };
    }
    let starts = linspace_int(train_end as i64, max_start as i64 - 1, n_windows);

    let mut errors = 0usize;
    let mut total_elapsed = std::time::Duration::ZERO;
    let mut valid_count = 0usize;

    for (i, &start) in starts.iter().enumerate() {
        if start + horizon > series.len() {
            errors += 1;
            continue;
        }
        let ctx_start = start.saturating_sub(context_len);
        if ctx_start >= start {
            errors += 1;
            continue;
        }
        let context = &series[ctx_start..start];
        let truth = &series[start..start + horizon];
        if context.len() < 2 {
            errors += 1;
            continue;
        }

        let t0 = Instant::now();
        match model.point_and_iqr(context, horizon) {
            Ok(pr) if pr.point.len() >= horizon => {
                total_elapsed += t0.elapsed();
                valid_count += 1;
                windows[i] = Some(WindowResult {
                    point: pr.point[..horizon].to_vec(),
                    truth: truth.to_vec(),
                    context: context.to_vec(),
                    iqr: pr.iqr.unwrap_or(f32::NAN),
                });
            }
            Ok(_) => errors += 1,
            Err(e) => {
                eprintln!("  [{model_name}] ERROR — {e}");
                errors += 1;
            }
        }
    }

    let mean_lat_ms = if valid_count > 0 {
        total_elapsed.as_secs_f64() / valid_count as f64 * 1000.0
    } else {
        f64::NAN
    };
    ModelRunResult {
        windows,
        mean_lat_ms,
        errors,
    }
}

/// `serde_json` serializes `f64::NAN` as JSON `null` (there's no NaN literal
/// in JSON) but can't deserialize `null` back into a plain `f64` — every
/// NaN-capable float field in the cache needs this to round-trip at all.
/// Metrics are legitimately NaN whenever a config produces zero valid
/// windows (e.g. a model whose native horizon is shorter than requested).
pub mod nan_null {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    #[serde(rename = "MAE", with = "nan_null")]
    pub mae: f64,
    #[serde(rename = "RMSE", with = "nan_null")]
    pub rmse: f64,
    #[serde(rename = "MASE", with = "nan_null")]
    pub mase: f64,
}

impl Metrics {
    pub const NAN: Metrics = Metrics {
        mae: f64::NAN,
        rmse: f64::NAN,
        mase: f64::NAN,
    };
}

fn mase_one(pred: &[f32], truth: &[f32], ctx: &[f32]) -> f64 {
    if ctx.len() < 2 {
        return f64::NAN;
    }
    let mut sum = 0.0f64;
    for w in ctx.windows(2) {
        sum += (w[1] - w[0]).abs() as f64;
    }
    let naive_mae = sum / (ctx.len() - 1) as f64;
    if naive_mae == 0.0 {
        return f64::NAN;
    }
    let n = pred.len().min(truth.len());
    let mae: f64 = (0..n)
        .map(|i| (pred[i] - truth[i]).abs() as f64)
        .sum::<f64>()
        / n as f64;
    mae / naive_mae
}

pub fn metrics_from_windows(windows: &[&WindowResult]) -> Metrics {
    if windows.is_empty() {
        return Metrics::NAN;
    }
    let mut maes = Vec::with_capacity(windows.len());
    let mut rmses = Vec::with_capacity(windows.len());
    let mut mases = Vec::with_capacity(windows.len());
    for w in windows {
        let n = w.point.len().min(w.truth.len());
        let mae: f64 = (0..n)
            .map(|i| (w.point[i] - w.truth[i]).abs() as f64)
            .sum::<f64>()
            / n as f64;
        let mse: f64 = (0..n)
            .map(|i| ((w.point[i] - w.truth[i]) as f64).powi(2))
            .sum::<f64>()
            / n as f64;
        maes.push(mae);
        rmses.push(mse.sqrt());
        let m = mase_one(&w.point, &w.truth, &w.context);
        if !m.is_nan() {
            mases.push(m);
        }
    }
    Metrics {
        mae: maes.iter().sum::<f64>() / maes.len() as f64,
        rmse: rmses.iter().sum::<f64>() / rmses.len() as f64,
        mase: if mases.is_empty() {
            f64::NAN
        } else {
            mases.iter().sum::<f64>() / mases.len() as f64
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_int_matches_numpy() {
        // np.linspace(0, 9, 5, dtype=int) == [0, 2, 4, 6, 9]
        assert_eq!(linspace_int(0, 9, 5), vec![0, 2, 4, 6, 9]);
        // np.linspace(5, 5, 1, dtype=int) == [5]
        assert_eq!(linspace_int(5, 5, 1), vec![5]);
        // np.linspace(3, 3, 4, dtype=int) == [3, 3, 3, 3]
        assert_eq!(linspace_int(3, 3, 4), vec![3, 3, 3, 3]);
    }

    #[test]
    fn linspace_int_empty() {
        assert!(linspace_int(0, 10, 0).is_empty());
    }

    fn window(point: &[f32], truth: &[f32], context: &[f32]) -> WindowResult {
        WindowResult {
            point: point.to_vec(),
            truth: truth.to_vec(),
            context: context.to_vec(),
            iqr: f32::NAN,
        }
    }

    #[test]
    fn metrics_perfect_prediction_is_zero_error() {
        let w = window(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0]);
        let m = metrics_from_windows(&[&w]);
        assert!((m.mae - 0.0).abs() < 1e-9);
        assert!((m.rmse - 0.0).abs() < 1e-9);
        assert!((m.mase - 0.0).abs() < 1e-9);
    }

    #[test]
    fn metrics_known_mae_rmse() {
        // |1-2| = 1, |2-4| = 2 -> MAE = 1.5, RMSE = sqrt((1+4)/2) = sqrt(2.5)
        let w = window(&[1.0, 2.0], &[2.0, 4.0], &[10.0, 11.0, 12.0]);
        let m = metrics_from_windows(&[&w]);
        assert!((m.mae - 1.5).abs() < 1e-9);
        assert!((m.rmse - 2.5f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn metrics_empty_windows_is_nan() {
        let m = metrics_from_windows(&[]);
        assert!(m.mae.is_nan() && m.rmse.is_nan() && m.mase.is_nan());
    }

    #[test]
    fn nan_null_round_trips_through_json() {
        let m = Metrics {
            mae: f64::NAN,
            rmse: 1.5,
            mase: f64::NAN,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            json.contains("null"),
            "NaN should serialize as JSON null, got {json}"
        );
        let back: Metrics = serde_json::from_str(&json).unwrap();
        assert!(back.mae.is_nan());
        assert!((back.rmse - 1.5).abs() < 1e-9);
        assert!(back.mase.is_nan());
    }
}
