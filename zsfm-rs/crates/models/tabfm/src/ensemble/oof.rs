//! Out-of-fold (OOF) prediction generation, used to fit calibration/NNLS. Fold splitting uses
//! our own seeded shuffle (`PyRandom`) rather than sklearn's `KFold(shuffle=True)`, which draws
//! from NumPy's legacy `RandomState` — a related but distinct RNG family we haven't ported (see
//! the plan's scoping note). Statistically equivalent, not bit-identical.

use anyhow::Result;
use serde_json::Value;

use crate::infer::TabFMModel;

use super::config_gen::MemberConfig;
use super::orchestrate::{run_members_classification, run_members_regression, EnsembleParams};
use super::pyrandom::PyRandom;

pub struct KFoldSplit {
    pub train_idx: Vec<usize>,
    pub val_idx: Vec<usize>,
}

/// A shuffled K-fold split of `0..n`.
pub fn kfold_splits(n: usize, k: usize, random_state: u64) -> Vec<KFoldSplit> {
    let mut indices: Vec<usize> = (0..n).collect();
    PyRandom::new(random_state).shuffle(&mut indices);

    let k = k.min(n).max(1);
    let base = n / k;
    let remainder = n % k;
    let mut folds = Vec::with_capacity(k);
    let mut start = 0;
    for i in 0..k {
        let size = base + if i < remainder { 1 } else { 0 };
        let val_idx: Vec<usize> = indices[start..start + size].to_vec();
        let train_idx: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(pos, _)| *pos < start || *pos >= start + size)
            .map(|(_, &v)| v)
            .collect();
        folds.push(KFoldSplit { train_idx, val_idx });
        start += size;
    }
    folds
}

fn select_rows<T: Clone>(rows: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| rows[i].clone()).collect()
}

/// Runs the full `n_estimators`-member ensemble on each of `num_folds` folds (fold's validation
/// rows as query, the rest as context), assembling `[member][original_train_row][class]`
/// un-shifted OOF logits (every training row appears in exactly one fold's validation set).
///
/// A flattened-across-folds version (grouping folds by `(train_size, val_size)` shape and running
/// all fold×member tasks through one `rayon` pass per shape group) was tried here, on the theory
/// that 5 sequential per-fold rounds (default `num_folds_for_cv=5`) waste parallelism when each
/// round only chunks `n_estimators` members into a handful of `rayon` chunks. Measured worse in
/// every configuration tried (same or ~25% slower with default chunking, ~75% slower forcing one
/// batch per shape group) — each fold's own `predict_batch` call already saturates Accelerate's
/// internal BLAS threading on this machine, so adding a `rayon` layer across folds/shape-groups on
/// top oversubscribes rather than helping (the same class of issue Round 1 documented for thread
/// count). Reverted; kept simple.
pub fn run_oof_classification(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_codes: &[f64],
    cat_mask: &[bool],
    n_classes: usize,
    configs: &[MemberConfig],
    p: &EnsembleParams,
    num_folds: usize,
) -> Result<Vec<Vec<Vec<f64>>>> {
    let n = x_train_raw.len();
    let folds = kfold_splits(n, num_folds, p.random_state);

    let mut oof: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n]; configs.len()];
    for fold in &folds {
        let fold_x_train = select_rows(x_train_raw, &fold.train_idx);
        let fold_y_train: Vec<f64> = fold.train_idx.iter().map(|&i| y_train_codes[i]).collect();
        let fold_x_val = select_rows(x_train_raw, &fold.val_idx);

        let per_member = run_members_classification(
            model, &fold_x_train, &fold_y_train, &fold_x_val, cat_mask, n_classes, configs, p.outlier_threshold,
            p.batch_size,
        )?;
        for (m, member_out) in per_member.into_iter().enumerate() {
            for (local_idx, &orig_idx) in fold.val_idx.iter().enumerate() {
                oof[m][orig_idx] = member_out[local_idx].clone();
            }
        }
    }
    Ok(oof)
}

/// Regression counterpart: `[member][original_train_row]` *scaled* OOF predictions.
pub fn run_oof_regression(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_scaled: &[f64],
    cat_mask: &[bool],
    configs: &[MemberConfig],
    p: &EnsembleParams,
    num_folds: usize,
) -> Result<Vec<Vec<f64>>> {
    let n = x_train_raw.len();
    let folds = kfold_splits(n, num_folds, p.random_state);

    let mut oof: Vec<Vec<f64>> = vec![vec![0.0; n]; configs.len()];
    for fold in &folds {
        let fold_x_train = select_rows(x_train_raw, &fold.train_idx);
        let fold_y_train: Vec<f64> = fold.train_idx.iter().map(|&i| y_train_scaled[i]).collect();
        let fold_x_val = select_rows(x_train_raw, &fold.val_idx);

        let per_member = run_members_regression(
            model, &fold_x_train, &fold_y_train, &fold_x_val, cat_mask, configs, p.outlier_threshold, p.batch_size,
        )?;
        for (m, member_out) in per_member.into_iter().enumerate() {
            for (local_idx, &orig_idx) in fold.val_idx.iter().enumerate() {
                oof[m][orig_idx] = member_out[local_idx];
            }
        }
    }
    Ok(oof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_covers_every_row_exactly_once() {
        let folds = kfold_splits(11, 5, 42);
        let mut seen = vec![0u32; 11];
        for f in &folds {
            for &i in &f.val_idx {
                seen[i] += 1;
            }
            assert!(!f.train_idx.is_empty());
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_kfold_fold_count_capped_by_n() {
        let folds = kfold_splits(3, 5, 42);
        assert_eq!(folds.len(), 3);
    }
}
