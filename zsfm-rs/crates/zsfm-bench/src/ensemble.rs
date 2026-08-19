use std::collections::HashMap;

use crate::eval::WindowResult;
use crate::models::ModelId;

pub const METHODS: &[(&str, &str)] = &[
    ("mean", "μ"),
    ("median", "~"),
    ("weighted", "w"),
    ("trim", "tr"),
    ("softmax", "sfx"),
    ("geometric", "geo"),
    ("online", "onl"),
    ("adaptive", "ada"),
    ("smooth", "smo"),
    ("select", "sel"),
    ("per_horiz", "wh"),
    ("uncertain", "uq"),
];

pub fn abbrev(method: &str) -> &'static str {
    METHODS
        .iter()
        .find(|(m, _)| *m == method)
        .map(|(_, a)| *a)
        .unwrap_or("?")
}

pub const WEIGHT_METHODS: &[&str] = &["weighted", "softmax"];
const SOFTMAX_TEMP: f64 = 0.5;
const SMO_ALPHA: f64 = 0.5;

pub fn label(combo: &[ModelId], method: &str) -> String {
    let parts: String = combo.iter().map(|m| m.letter()).collect();
    format!("{parts}({})", abbrev(method))
}

fn median_of(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

/// `preds[model][t]`. Stateless combination methods (mean/median/weighted/
/// trim/softmax/geometric), ported 1:1 from the old Python `_combine`.
fn combine(preds: &[Vec<f64>], method: &str, w: Option<&[f64]>) -> Vec<f64> {
    let n = preds.len();
    let horizon = preds[0].len();

    match method {
        "mean" => (0..horizon)
            .map(|t| preds.iter().map(|p| p[t]).sum::<f64>() / n as f64)
            .collect(),
        "median" => (0..horizon)
            .map(|t| {
                let mut col: Vec<f64> = preds.iter().map(|p| p[t]).collect();
                median_of(&mut col)
            })
            .collect(),
        "weighted" => {
            let w = w.expect("weighted needs weights");
            let sum: f64 = w.iter().sum();
            let wn: Vec<f64> = w.iter().map(|x| x / sum).collect();
            (0..horizon)
                .map(|t| preds.iter().zip(&wn).map(|(p, wi)| p[t] * wi).sum::<f64>())
                .collect()
        }
        "trim" => {
            if n <= 2 {
                combine(preds, "mean", None)
            } else {
                (0..horizon)
                    .map(|t| {
                        let mut col: Vec<f64> = preds.iter().map(|p| p[t]).collect();
                        col.sort_by(|a, b| a.total_cmp(b));
                        let trimmed = &col[1..col.len() - 1];
                        trimmed.iter().sum::<f64>() / trimmed.len() as f64
                    })
                    .collect()
            }
        }
        "softmax" => {
            let w = w.expect("softmax needs weights");
            let sum: f64 = w.iter().sum();
            let logits: Vec<f64> = w.iter().map(|x| x / sum / SOFTMAX_TEMP).collect();
            let max_l = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exp_w: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
            let s: f64 = exp_w.iter().sum();
            let wn: Vec<f64> = exp_w.iter().map(|x| x / s).collect();
            (0..horizon)
                .map(|t| preds.iter().zip(&wn).map(|(p, wi)| p[t] * wi).sum::<f64>())
                .collect()
        }
        "geometric" => (0..horizon)
            .map(|t| {
                let col: Vec<f64> = preds.iter().map(|p| p[t]).collect();
                let med = median_of(&mut col.clone());
                let sign = if med == 0.0 { 1.0 } else { med.signum() };
                let log_abs_mean: f64 =
                    col.iter().map(|v| (v.abs() + 1e-9).ln()).sum::<f64>() / col.len() as f64;
                sign * log_abs_mean.exp()
            })
            .collect(),
        _ => unreachable!("unknown stateless method {method}"),
    }
}

fn per_model_mae(preds: &[Vec<f64>], truth: &[f64]) -> Vec<f64> {
    preds
        .iter()
        .map(|p| p.iter().zip(truth).map(|(a, b)| (a - b).abs()).sum::<f64>() / truth.len() as f64)
        .collect()
}

fn clip_inv(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| 1.0 / x.max(1e-12)).collect()
}

/// Build ensemble windows for one model combo + strategy. Only positions
/// where every selected model produced a valid window are included. Online
/// strategies (`online`/`adaptive`/`smooth`/`select`/`per_horiz`/`uncertain`)
/// carry state forward window-by-window, in index order — causal, matching
/// the old Python implementation exactly.
pub fn ensemble_windows(
    all_windows: &HashMap<ModelId, Vec<Option<WindowResult>>>,
    combo: &[ModelId],
    method: &str,
    static_weights: Option<&HashMap<ModelId, f64>>,
) -> Vec<WindowResult> {
    let n = all_windows.values().next().map(|v| v.len()).unwrap_or(0);
    let n_models = combo.len();
    let mut result = Vec::new();

    let mut online_inv_mae = vec![1.0f64; n_models];

    let mut hedge_w = vec![1.0f64; n_models];
    let hedge_eta = (2.0 * (n_models.max(2) as f64).ln() / n.max(1) as f64).sqrt();

    let mut smo_inv_mae = vec![1.0f64; n_models];
    let mut sel_cum_mae = vec![1.0f64; n_models];
    let mut sel_n_valid = 0usize;
    let mut wh_inv_mae: Option<Vec<Vec<f64>>> = None;
    let mut uq_fallback_mae = vec![1.0f64; n_models];

    for i in 0..n {
        let entries: Option<Vec<&WindowResult>> = combo
            .iter()
            .map(|m| all_windows.get(m).and_then(|v| v[i].as_ref()))
            .collect();
        let Some(entries) = entries else { continue };

        let preds: Vec<Vec<f64>> = entries
            .iter()
            .map(|e| e.point.iter().map(|&v| v as f64).collect())
            .collect();
        let truth: Vec<f64> = entries[0].truth.iter().map(|&v| v as f64).collect();
        let context = entries[0].context.clone();
        let horizon = truth.len();

        let combined: Vec<f64> = match method {
            "online" => {
                let combined = combine(&preds, "weighted", Some(&online_inv_mae));
                let per_mae = per_model_mae(&preds, &truth);
                online_inv_mae = clip_inv(&per_mae);
                combined
            }
            "adaptive" => {
                let sum: f64 = hedge_w.iter().sum();
                let w: Vec<f64> = hedge_w.iter().map(|x| x / sum).collect();
                let combined = combine(&preds, "weighted", Some(&w));
                let losses = per_model_mae(&preds, &truth);
                for (hw, l) in hedge_w.iter_mut().zip(&losses) {
                    *hw *= (-hedge_eta * l).exp();
                }
                let s: f64 = hedge_w.iter().sum();
                for hw in hedge_w.iter_mut() {
                    *hw /= s;
                }
                combined
            }
            "smooth" => {
                let sum: f64 = smo_inv_mae.iter().sum();
                let w: Vec<f64> = smo_inv_mae.iter().map(|x| x / sum).collect();
                let combined = combine(&preds, "weighted", Some(&w));
                let per_mae = per_model_mae(&preds, &truth);
                let cur_inv = clip_inv(&per_mae);
                for (s, c) in smo_inv_mae.iter_mut().zip(&cur_inv) {
                    *s = SMO_ALPHA * c + (1.0 - SMO_ALPHA) * *s;
                }
                combined
            }
            "select" => {
                let combined = if sel_n_valid == 0 {
                    combine(&preds, "mean", None)
                } else {
                    let best = sel_cum_mae
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    preds[best].clone()
                };
                let per_mae = per_model_mae(&preds, &truth);
                for (s, m) in sel_cum_mae.iter_mut().zip(&per_mae) {
                    *s = (1.0 - SMO_ALPHA) * *s + SMO_ALPHA * m;
                }
                sel_n_valid += 1;
                combined
            }
            "per_horiz" => {
                let is_first = wh_inv_mae.is_none();
                let state = wh_inv_mae.get_or_insert_with(|| vec![vec![1.0f64; horizon]; n_models]);
                let combined = if is_first {
                    combine(&preds, "mean", None)
                } else {
                    // Column-normalize per horizon step across models.
                    let colsum: Vec<f64> = (0..horizon)
                        .map(|t| state.iter().map(|row| row[t]).sum::<f64>())
                        .collect();
                    (0..horizon)
                        .map(|t| {
                            preds
                                .iter()
                                .zip(state.iter())
                                .map(|(p, row)| p[t] * row[t] / colsum[t])
                                .sum::<f64>()
                        })
                        .collect()
                };
                let per_step_err: Vec<Vec<f64>> = preds
                    .iter()
                    .map(|p| (0..horizon).map(|t| (p[t] - truth[t]).abs()).collect())
                    .collect();
                let state = wh_inv_mae.as_mut().unwrap();
                for (row, err_row) in state.iter_mut().zip(per_step_err.iter()) {
                    for t in 0..horizon {
                        let step_inv = 1.0 / err_row[t].max(1e-12);
                        row[t] = SMO_ALPHA * step_inv + (1.0 - SMO_ALPHA) * row[t];
                    }
                }
                combined
            }
            "uncertain" => {
                let unc_vals: Vec<f64> = entries
                    .iter()
                    .enumerate()
                    .map(|(j, e)| {
                        if e.iqr.is_finite() {
                            e.iqr as f64
                        } else {
                            uq_fallback_mae[j]
                        }
                    })
                    .collect();
                let w = clip_inv(&unc_vals);
                let combined = combine(&preds, "weighted", Some(&w));
                uq_fallback_mae = per_model_mae(&preds, &truth);
                combined
            }
            _ => {
                let w: Option<Vec<f64>> = if WEIGHT_METHODS.contains(&method) {
                    Some(
                        combo
                            .iter()
                            .map(|m| {
                                static_weights
                                    .and_then(|sw| sw.get(m))
                                    .copied()
                                    .unwrap_or(1e-9)
                            })
                            .collect(),
                    )
                } else {
                    None
                };
                combine(&preds, method, w.as_deref())
            }
        };

        result.push(WindowResult {
            point: combined.into_iter().map(|v| v as f32).collect(),
            truth: entries[0].truth.clone(),
            context,
            iqr: f32::NAN,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_of_two_models() {
        let preds = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(combine(&preds, "mean", None), vec![2.0, 3.0]);
    }

    #[test]
    fn median_of_three_models_picks_middle() {
        let preds = vec![vec![1.0], vec![5.0], vec![3.0]];
        assert_eq!(combine(&preds, "median", None), vec![3.0]);
    }

    #[test]
    fn weighted_matches_hand_computed() {
        // weights [1, 3] normalize to [0.25, 0.75]
        let preds = vec![vec![0.0], vec![4.0]];
        let out = combine(&preds, "weighted", Some(&[1.0, 3.0]));
        assert!((out[0] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn trim_drops_min_and_max() {
        // 4 models per timestep: [1, 2, 3, 100] -> drop 1 and 100, mean(2,3)=2.5
        let preds = vec![vec![1.0], vec![2.0], vec![3.0], vec![100.0]];
        let out = combine(&preds, "trim", None);
        assert!((out[0] - 2.5).abs() < 1e-9);
    }

    #[test]
    fn trim_falls_back_to_mean_below_three_models() {
        let preds = vec![vec![2.0], vec![4.0]];
        assert_eq!(combine(&preds, "trim", None), combine(&preds, "mean", None));
    }

    #[test]
    fn geometric_preserves_sign() {
        let preds = vec![vec![-2.0], vec![-8.0]];
        let out = combine(&preds, "geometric", None);
        assert!(out[0] < 0.0, "expected negative result, got {out:?}");
    }

    #[test]
    fn label_formats_combo_and_method() {
        assert_eq!(
            label(&[ModelId::Toto, ModelId::Chronos], "weighted"),
            "TC(w)"
        );
    }
}
