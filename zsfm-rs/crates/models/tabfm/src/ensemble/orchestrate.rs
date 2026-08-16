//! Orchestrates one full `ensemble-predict` call: builds the `n_estimators` member configs,
//! runs each member's preprocessing + a single `TabFMModel::predict` forward pass (reusing the
//! core model unchanged), then aggregates — optionally applying calibration/NNLS ensemble
//! weighting fit via `oof.rs`'s out-of-fold procedure.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::Value;

use crate::infer::TabFMModel;

use super::aggregate::{self, ClassAggMode};
use super::calibration::{PlattParams, VectorScalingParams};
use super::cat_encoder::{CategoricalOrdinalEncoder, LabelEncoder};
use super::config_gen::{self, EnsembleConfigParams, MemberConfig, NormMethod};
use super::nnls;
use super::oof;
use super::scalers::{self, StandardScaler};

/// Default members-per-batch when `EnsembleParams::batch_size` is `None`, chosen from
/// benchmarking on an Apple M4 across small/medium table sizes — see README's Performance
/// section. One giant batch of all `n_estimators` measured *slower* than this in every case
/// tried (attention cost scales with `B*T^2`, so a single huge batch loses `rayon` parallelism
/// without a compensating BLAS efficiency win).
const DEFAULT_BATCH_CHUNK_SIZE: usize = 8;

/// Parameters for one `ensemble-predict` call — the sklearn-wrapper-equivalent counterpart to
/// `TabFMClassifier(...)`/`TabFMRegressor(...)`'s constructor kwargs. Fields are private; build
/// one by chaining `.with_*()` off [`EnsembleParams::default`] (defaults match the wrapper's
/// own constructor defaults).
///
/// ```
/// use zsfm_tabfm::EnsembleParams;
///
/// let params = EnsembleParams::default()
///     .with_n_estimators(64)
///     .with_enable_nnls(true)
///     .with_random_state(0);
/// ```
pub struct EnsembleParams {
    pub(crate) n_estimators: usize,
    pub(crate) norm_methods: Vec<NormMethod>,
    pub(crate) class_shift: bool,
    pub(crate) outlier_threshold: f64,
    pub(crate) softmax_temperature: f64,
    pub(crate) average_logits: bool,
    pub(crate) random_state: u64,
    pub(crate) binary_calibration: bool,  // "platt"
    pub(crate) multiclass_calibration: bool, // "vector"
    pub(crate) num_folds_for_cv: usize,
    pub(crate) enable_nnls: bool,
    pub(crate) nnls_beta: f64,
    pub(crate) calibration_lambda: f64,
    /// Members per `TabFMModel::predict_batch` call. `None` (default) chunks members into groups
    /// of `DEFAULT_BATCH_CHUNK_SIZE`, run in parallel via `rayon` — measured faster than one giant
    /// batch of all `n_estimators` (attention cost scales with `B*T^2`, so a single huge batch
    /// doesn't get the BLAS efficiency win a batched *linear* layer would, while still giving up
    /// `rayon` parallelism). Tune for a given table size/machine (see README's Performance
    /// section); `Some(n_estimators)` recovers the single-giant-batch behavior if desired.
    pub(crate) batch_size: Option<usize>,
}

impl Default for EnsembleParams {
    fn default() -> Self {
        EnsembleParams {
            n_estimators: 32,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            class_shift: true,
            outlier_threshold: 4.0,
            softmax_temperature: 0.9,
            average_logits: true,
            random_state: 42,
            binary_calibration: false,
            multiclass_calibration: false,
            num_folds_for_cv: 5,
            enable_nnls: false,
            nnls_beta: 0.75,
            calibration_lambda: 1e-2,
            batch_size: None,
        }
    }
}

impl EnsembleParams {
    pub fn with_n_estimators(mut self, v: usize) -> Self { self.n_estimators = v; self }
    pub fn with_norm_methods(mut self, v: Vec<NormMethod>) -> Self { self.norm_methods = v; self }
    pub fn with_class_shift(mut self, v: bool) -> Self { self.class_shift = v; self }
    pub fn with_outlier_threshold(mut self, v: f64) -> Self { self.outlier_threshold = v; self }
    pub fn with_softmax_temperature(mut self, v: f64) -> Self { self.softmax_temperature = v; self }
    pub fn with_average_logits(mut self, v: bool) -> Self { self.average_logits = v; self }
    pub fn with_random_state(mut self, v: u64) -> Self { self.random_state = v; self }
    /// Enable Platt scaling for binary classification (`n_classes == 2`).
    pub fn with_binary_calibration(mut self, v: bool) -> Self { self.binary_calibration = v; self }
    /// Enable vector scaling for multiclass classification (`n_classes > 2`).
    pub fn with_multiclass_calibration(mut self, v: bool) -> Self { self.multiclass_calibration = v; self }
    pub fn with_num_folds_for_cv(mut self, v: usize) -> Self { self.num_folds_for_cv = v; self }
    pub fn with_enable_nnls(mut self, v: bool) -> Self { self.enable_nnls = v; self }
    pub fn with_nnls_beta(mut self, v: f64) -> Self { self.nnls_beta = v; self }
    pub fn with_calibration_lambda(mut self, v: f64) -> Self { self.calibration_lambda = v; self }
    pub fn with_batch_size(mut self, v: Option<usize>) -> Self { self.batch_size = v; self }
}

pub struct ClassificationOutput {
    pub probabilities: Vec<Vec<f64>>, // [n_test][n_classes], class order = LabelEncoder order
    pub predicted_labels: Vec<String>,
    pub classes: Vec<String>,
}

pub struct RegressionOutput {
    pub predictions: Vec<f64>, // [n_test]
}

/// One member's preprocessed table, ready for `TabFMModel::predict`.
struct MemberTable {
    x: Vec<Vec<f32>>, // [n_train+n_test][n_features]
    cat_mask: Vec<bool>,
}

/// Per-original-column preprocessing, precomputed **once** and shared read-only across all
/// ensemble members — a member's feature permutation only changes which *position* a column
/// lands in, and its `norm_method` only changes which of the (few, cycled) normalizer variants is
/// used, neither of which changes a column's own encoded values or a given normalizer's fit on
/// them. Without this, `CategoricalOrdinalEncoder::fit` and `scalers::apply_pipeline` (including
/// `PowerTransformer`'s iterative Brent's-method optimization) would needlessly re-run once per
/// member (up to `n_estimators` times) instead of once per `(column, norm_method)` pair (at most
/// `norm_methods.len()` times).
struct ColumnCache {
    /// `(column, norm_method) -> (scaled_train, scaled_query)`, precomputed for every norm
    /// method actually used by `configs`.
    scaled: HashMap<(usize, NormMethod), (Vec<f64>, Vec<f64>)>,
}

impl ColumnCache {
    fn build(
        x_train_raw: &[Vec<Value>],
        x_query_raw: &[Vec<Value>],
        cat_mask: &[bool],
        configs: &[MemberConfig],
        outlier_threshold: f64,
    ) -> Self {
        let n_features = x_train_raw.first().map(|r| r.len()).unwrap_or(0);

        let mut raw_train: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        let mut raw_query: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for col in 0..n_features {
            let train_raw: Vec<Value> = x_train_raw.iter().map(|r| r[col].clone()).collect();
            let query_raw: Vec<Value> = x_query_raw.iter().map(|r| r[col].clone()).collect();
            let (t, q) = if cat_mask[col] {
                let enc = CategoricalOrdinalEncoder::fit(&train_raw);
                (enc.transform(&train_raw), enc.transform(&query_raw))
            } else {
                let parse = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
                (train_raw.iter().map(parse).collect(), query_raw.iter().map(parse).collect())
            };
            raw_train.push(t);
            raw_query.push(q);
        }

        let distinct_norm_methods: HashSet<NormMethod> = configs.iter().map(|c| c.norm_method).collect();

        let mut scaled = HashMap::with_capacity(n_features * distinct_norm_methods.len());
        for col in 0..n_features {
            for &nm in &distinct_norm_methods {
                let transformed = scalers::apply_pipeline(&raw_train[col], &raw_query[col], nm, outlier_threshold);
                scaled.insert((col, nm), transformed);
            }
        }

        ColumnCache { scaled }
    }

    fn scaled(&self, col: usize, norm_method: NormMethod) -> (&[f64], &[f64]) {
        let (t, q) = self.scaled.get(&(col, norm_method)).expect("precomputed for every config's norm_method");
        (t, q)
    }
}

/// Assembles one member's table by looking up its columns (in permuted order) from the shared
/// `ColumnCache` — no per-member encoding/scaling work left, just a gather + transpose.
fn build_member_table(
    cache: &ColumnCache,
    n_train: usize,
    n_query: usize,
    cat_mask: &[bool],
    member: &MemberConfig,
) -> MemberTable {
    let h = member.feature_permutation.len();
    let mut x = vec![vec![0f32; h]; n_train + n_query];
    let mut permuted_cat_mask: Vec<bool> = Vec::with_capacity(h);

    for (pos, &src_col) in member.feature_permutation.iter().enumerate() {
        permuted_cat_mask.push(cat_mask[src_col]);
        let (train_scaled, query_scaled) = cache.scaled(src_col, member.norm_method);
        for (row_idx, &v) in train_scaled.iter().enumerate() {
            x[row_idx][pos] = v as f32;
        }
        for (row_idx, &v) in query_scaled.iter().enumerate() {
            x[n_train + row_idx][pos] = v as f32;
        }
    }

    MemberTable { x, cat_mask: permuted_cat_mask }
}

/// Runs every ensemble member's forward pass for classification, given a train/query row split
/// (query rows may be real held-out test rows, or an OOF fold's validation rows). Members are
/// grouped into `batch_size`-sized chunks (default: `DEFAULT_BATCH_CHUNK_SIZE`) and each chunk
/// runs as a **single** `TabFMModel::predict_batch` call — sharing the fixed cost of the model's
/// deepest stage (24-block ICL) across the whole chunk instead of paying it once per member.
/// Chunks run in parallel via `rayon`, combining with Round 1's parallelism.
/// Returns `[member][query_row][class]` logits, already un-shifted back to original class order.
#[allow(clippy::too_many_arguments)]
pub fn run_members_classification(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_codes: &[f64],
    x_query_raw: &[Vec<Value>],
    cat_mask: &[bool],
    n_classes: usize,
    configs: &[MemberConfig],
    outlier_threshold: f64,
    batch_size: Option<usize>,
) -> Result<Vec<Vec<Vec<f64>>>> {
    let n_train = x_train_raw.len();
    let n_query = x_query_raw.len();
    let n_features = x_train_raw.first().map(|r| r.len()).unwrap_or(0);

    let cache = ColumnCache::build(x_train_raw, x_query_raw, cat_mask, configs, outlier_threshold);
    let tables: Vec<MemberTable> =
        configs.iter().map(|member| build_member_table(&cache, n_train, n_query, cat_mask, member)).collect();

    let chunk_size = batch_size.unwrap_or(DEFAULT_BATCH_CHUNK_SIZE).max(1);
    let num_chunks = configs.len().div_ceil(chunk_size);

    let chunks: Result<Vec<Vec<Vec<Vec<f64>>>>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| -> Result<Vec<Vec<Vec<f64>>>> {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(configs.len());
            let member_chunk = &configs[start..end];
            let table_chunk = &tables[start..end];

            let x_batch: Vec<Vec<Vec<f32>>> = table_chunk.iter().map(|t| t.x.clone()).collect();
            let cat_mask_batch: Vec<Vec<bool>> = table_chunk.iter().map(|t| t.cat_mask.clone()).collect();
            let y_batch: Vec<Vec<f32>> = member_chunk
                .iter()
                .map(|member| {
                    y_train_codes
                        .iter()
                        .map(|&c| ((c as i64 + member.class_shift as i64).rem_euclid(n_classes as i64)) as f32)
                        .chain(std::iter::repeat(0f32).take(n_query))
                        .collect()
                })
                .collect();

            let out_batch = model
                .predict_batch(&x_batch, &y_batch, n_train, &cat_mask_batch, Some(n_features))
                .context("member batch forward pass")?;

            Ok(member_chunk
                .iter()
                .zip(out_batch.iter())
                .map(|(member, out)| {
                    out[n_train..]
                        .iter()
                        .map(|row| {
                            let row64: Vec<f64> = row.iter().map(|&v| v as f64).collect();
                            aggregate::unshift_logits(&row64, member.class_shift, n_classes)
                        })
                        .collect()
                })
                .collect())
        })
        .collect();

    Ok(chunks?.into_iter().flatten().collect())
}

/// Same idea for regression: returns `[member][query_row]` *scaled* (not yet inverse-transformed)
/// predictions.
pub fn run_members_regression(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_scaled: &[f64],
    x_query_raw: &[Vec<Value>],
    cat_mask: &[bool],
    configs: &[MemberConfig],
    outlier_threshold: f64,
    batch_size: Option<usize>,
) -> Result<Vec<Vec<f64>>> {
    let n_train = x_train_raw.len();
    let n_query = x_query_raw.len();
    let n_features = x_train_raw.first().map(|r| r.len()).unwrap_or(0);

    let cache = ColumnCache::build(x_train_raw, x_query_raw, cat_mask, configs, outlier_threshold);
    let tables: Vec<MemberTable> =
        configs.iter().map(|member| build_member_table(&cache, n_train, n_query, cat_mask, member)).collect();

    let chunk_size = batch_size.unwrap_or(DEFAULT_BATCH_CHUNK_SIZE).max(1);
    let num_chunks = configs.len().div_ceil(chunk_size);

    let chunks: Result<Vec<Vec<Vec<f64>>>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| -> Result<Vec<Vec<f64>>> {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(configs.len());
            let table_chunk = &tables[start..end];

            let x_batch: Vec<Vec<Vec<f32>>> = table_chunk.iter().map(|t| t.x.clone()).collect();
            let cat_mask_batch: Vec<Vec<bool>> = table_chunk.iter().map(|t| t.cat_mask.clone()).collect();
            let y_batch: Vec<Vec<f32>> = (0..table_chunk.len())
                .map(|_| y_train_scaled.iter().map(|&v| v as f32).chain(std::iter::repeat(0f32).take(n_query)).collect())
                .collect();

            let out_batch = model
                .predict_batch(&x_batch, &y_batch, n_train, &cat_mask_batch, Some(n_features))
                .context("member batch forward pass")?;

            Ok(out_batch.iter().map(|out| out[n_train..].iter().map(|row| row[0] as f64).collect()).collect())
        })
        .collect();

    Ok(chunks?.into_iter().flatten().collect())
}

fn class_agg_mode<'a>(p: &EnsembleParams, nnls_weights: &'a Option<Vec<f64>>) -> ClassAggMode<'a> {
    match nnls_weights {
        Some(w) => ClassAggMode::NnlsWeighted(w),
        None if p.average_logits => ClassAggMode::AverageLogits,
        None => ClassAggMode::AverageProbs,
    }
}

pub fn run_classification(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_raw: &[Value],
    x_test_raw: &[Vec<Value>],
    cat_mask: &[bool],
    p: &EnsembleParams,
) -> Result<ClassificationOutput> {
    let n_train = x_train_raw.len();
    let n_test = x_test_raw.len();
    let n_features = x_train_raw.first().map(|r| r.len()).unwrap_or(0);

    let label_enc = LabelEncoder::fit(y_train_raw);
    let n_classes = label_enc.n_classes();
    let y_codes = label_enc.transform(y_train_raw);

    let configs = config_gen::generate_ensemble(&EnsembleConfigParams {
        n_estimators: p.n_estimators,
        n_features,
        n_train_rows: n_train,
        is_classification: true,
        n_classes,
        class_shift: p.class_shift,
        permute_categorical: false,
        cat_value_counts: vec![],
        max_num_rows: None,
        norm_methods: p.norm_methods.clone(),
        random_state: p.random_state,
    });

    let per_member_logits = run_members_classification(
        model, x_train_raw, &y_codes, x_test_raw, cat_mask, n_classes, &configs, p.outlier_threshold, p.batch_size,
    )?;

    let binary_or_multiclass_calibration = if n_classes == 2 { p.binary_calibration } else { p.multiclass_calibration };

    // Compute out-of-fold logits at most once, regardless of how many of {NNLS, calibration} are
    // requested — both need the same OOF ensemble run, and each is itself a `num_folds_for_cv` x
    // `n_estimators` re-run of the whole forward pass, so this reuse matters a lot in practice.
    let oof_logits = if p.enable_nnls || binary_or_multiclass_calibration {
        Some(oof::run_oof_classification(
            model, x_train_raw, &y_codes, cat_mask, n_classes, &configs, p, p.num_folds_for_cv,
        )?)
    } else {
        None
    };

    let nnls_weights = if p.enable_nnls {
        let oof_logits = oof_logits.as_ref().expect("computed above when enable_nnls");
        // oof_logits: [member][train_row][class] -> per-member OOF probabilities for NNLS.
        let oof_probs: Vec<Vec<Vec<f64>>> = oof_logits
            .iter()
            .map(|m| m.iter().map(|l| aggregate::softmax_temperature(l, p.softmax_temperature)).collect())
            .collect();
        let n_est = oof_probs.len();
        let n_tr = oof_probs[0].len();
        let mut design: Vec<Vec<f64>> = vec![vec![0.0; n_tr * n_classes]; n_est];
        let mut target = vec![0.0; n_tr * n_classes];
        for (r, &yc) in y_codes.iter().enumerate() {
            target[r * n_classes + yc as usize] = 1.0;
        }
        for (m, member_probs) in oof_probs.iter().enumerate() {
            for (r, probs) in member_probs.iter().enumerate() {
                for c in 0..n_classes {
                    design[m][r * n_classes + c] = probs[c];
                }
            }
        }
        let raw_weights = nnls::nnls(&design, &target);
        Some(nnls::finalize_weights(&raw_weights, p.nnls_beta))
    } else {
        None
    };

    let mode = class_agg_mode(p, &nnls_weights);
    let mut probabilities = Vec::with_capacity(n_test);
    for row_idx in 0..n_test {
        let logits_all: Vec<Vec<f64>> = per_member_logits.iter().map(|m| m[row_idx].clone()).collect();
        probabilities.push(aggregate::aggregate_classification(&logits_all, p.softmax_temperature, &mode));
    }

    if binary_or_multiclass_calibration {
        // Reuse the OOF logits computed above (shared with NNLS when both are enabled) and
        // aggregate them the *same* way (NNLS-or-average) test predictions were, to calibrate
        // the ensemble's actual output distribution, then apply the fitted transform to test preds.
        let oof_logits = oof_logits.as_ref().expect("computed above when calibration enabled");
        let n_tr = oof_logits[0].len();
        let mut oof_final_probs = Vec::with_capacity(n_tr);
        for row_idx in 0..n_tr {
            let logits_all: Vec<Vec<f64>> = oof_logits.iter().map(|m| m[row_idx].clone()).collect();
            oof_final_probs.push(aggregate::aggregate_classification(&logits_all, p.softmax_temperature, &mode));
        }
        let y_codes_usize: Vec<usize> = y_codes.iter().map(|&c| c as usize).collect();

        if n_classes == 2 {
            let params = PlattParams::fit(&oof_final_probs, &y_codes_usize, p.calibration_lambda);
            probabilities = probabilities.iter().map(|p| params.apply(p)).collect();
        } else {
            let params = VectorScalingParams::fit(&oof_final_probs, &y_codes_usize, p.calibration_lambda);
            probabilities = probabilities.iter().map(|p| params.apply(p)).collect();
        }
    }

    let mut predicted_labels = Vec::with_capacity(n_test);
    for probs in &probabilities {
        let (best_idx, _) =
            probs.iter().enumerate().fold((0, f64::NEG_INFINITY), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        predicted_labels.push(label_enc.decode(best_idx).to_string());
    }

    let classes: Vec<String> = (0..n_classes).map(|c| label_enc.decode(c).to_string()).collect();
    Ok(ClassificationOutput { probabilities, predicted_labels, classes })
}

pub fn run_regression(
    model: &TabFMModel,
    x_train_raw: &[Vec<Value>],
    y_train_raw: &[f64],
    x_test_raw: &[Vec<Value>],
    cat_mask: &[bool],
    p: &EnsembleParams,
) -> Result<RegressionOutput> {
    let n_train = x_train_raw.len();
    let n_features = x_train_raw.first().map(|r| r.len()).unwrap_or(0);

    let y_scaler = StandardScaler::fit(y_train_raw);
    let y_scaled = y_scaler.transform(y_train_raw);

    let configs = config_gen::generate_ensemble(&EnsembleConfigParams {
        n_estimators: p.n_estimators,
        n_features,
        n_train_rows: n_train,
        is_classification: false,
        n_classes: 1,
        class_shift: false,
        permute_categorical: false,
        cat_value_counts: vec![],
        max_num_rows: None,
        norm_methods: p.norm_methods.clone(),
        random_state: p.random_state,
    });

    let per_member_scaled_preds = run_members_regression(
        model, x_train_raw, &y_scaled, x_test_raw, cat_mask, &configs, p.outlier_threshold, p.batch_size,
    )?;

    let nnls_weights = if p.enable_nnls {
        let oof_scaled = oof::run_oof_regression(model, x_train_raw, &y_scaled, cat_mask, &configs, p, p.num_folds_for_cv)?;
        let n_est = oof_scaled.len();
        let n_tr = oof_scaled[0].len();
        // NNLS target uses inverse-transformed (original-scale) OOF predictions vs raw y.
        let mut design: Vec<Vec<f64>> = vec![vec![0.0; n_tr]; n_est];
        for m in 0..n_est {
            for r in 0..n_tr {
                design[m][r] = y_scaler.inverse_transform_scalar(oof_scaled[m][r]);
            }
        }
        let raw_weights = nnls::nnls(&design, y_train_raw);
        Some(nnls::finalize_weights(&raw_weights, p.nnls_beta))
    } else {
        None
    };

    let n_test = x_test_raw.len();
    let mut predictions = Vec::with_capacity(n_test);
    for row_idx in 0..n_test {
        let pred = match &nnls_weights {
            Some(weights) => {
                let unscaled: Vec<f64> = per_member_scaled_preds
                    .iter()
                    .map(|m| y_scaler.inverse_transform_scalar(m[row_idx]))
                    .collect();
                aggregate::weighted_unscaled_predictions(&unscaled, weights)
            }
            None => {
                let scaled_across_members: Vec<f64> = per_member_scaled_preds.iter().map(|m| m[row_idx]).collect();
                let avg_scaled = aggregate::average_scaled_predictions(&scaled_across_members);
                y_scaler.inverse_transform_scalar(avg_scaled)
            }
        };
        predictions.push(pred);
    }

    Ok(RegressionOutput { predictions })
}
