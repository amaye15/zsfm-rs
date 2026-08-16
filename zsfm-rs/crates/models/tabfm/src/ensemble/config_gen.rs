//! Ports `TabFMClassifier`/`TabFMRegressor`'s `_generate_ensemble()` — builds the `n_estimators`
//! member configs (feature permutation, classification class-shift offset, categorical-value
//! permutation, row-subsample pattern, normalization method). RNG consumption order matches the
//! source exactly (see the plan doc / module comments below) so results are bit-identical to the
//! real wrapper for the same `random_state`.

use super::pyrandom::PyRandom;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormMethod {
    None,
    Power,
    Quantile,
    QuantileRtdl,
    Robust,
}

impl NormMethod {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "none" => NormMethod::None,
            "power" => NormMethod::Power,
            "quantile" => NormMethod::Quantile,
            "quantile_rtdl" => NormMethod::QuantileRtdl,
            "robust" => NormMethod::Robust,
            other => anyhow::bail!("unknown norm_method: {other}"),
        })
    }
}

pub struct MemberConfig {
    /// Permutation of `0..n_features` (or a subsampled+permuted subset if
    /// `n_features > max_num_features`).
    pub feature_permutation: Vec<usize>,
    /// Amount added (mod `n_classes`) to every training label fed to the model for this member.
    /// Always `0` for regression or when class-shift is disabled/inapplicable.
    pub class_shift: usize,
    /// `cat_col_index -> (original_code -> permuted_code)`, present only when
    /// `permute_categorical=true` (default: off, so this is `None` for every member).
    pub cat_permutation: Option<Vec<Vec<usize>>>,
    /// Row indices to bag from `0..n_train`, present only when row-subsampling is active
    /// (default: off, so this is `None` for every member — use all rows).
    pub row_subsample: Option<Vec<usize>>,
    pub norm_method: NormMethod,
}

pub struct EnsembleConfigParams {
    pub n_estimators: usize,
    pub n_features: usize,
    pub n_train_rows: usize,
    pub is_classification: bool,
    pub n_classes: usize,
    pub class_shift: bool,
    pub permute_categorical: bool,
    /// Number of distinct values per categorical column (only consulted if
    /// `permute_categorical` is true); empty if there are no categorical columns.
    pub cat_value_counts: Vec<usize>,
    pub max_num_rows: Option<usize>,
    pub norm_methods: Vec<NormMethod>,
    pub random_state: u64,
}

/// Feature-permutation generation (`FeatureShuffler`): its own independent `random.Random`
/// stream, seeded with the same `random_state` but never mixed with the main RNG below.
fn generate_feature_permutations(n_features: usize, n_estimators: usize, random_state: u64) -> Vec<Vec<usize>> {
    let mut rng = PyRandom::new(random_state);
    if n_features <= 5 {
        let all_perms = permutations(n_features);
        let k = n_estimators.min(all_perms.len());
        rng.sample(&all_perms, k)
    } else {
        (0..n_estimators).map(|_| rng.sample_indices(n_features, n_features)).collect()
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    let mut items: Vec<usize> = (0..n).collect();
    let mut result = Vec::new();
    permute_rec(&mut items, 0, &mut result);
    result
}

fn permute_rec(items: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == items.len() {
        out.push(items.clone());
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permute_rec(items, k + 1, out);
        items.swap(k, i);
    }
}

/// The full `_generate_ensemble()` port. See module docs for the exact RNG call order this
/// must preserve: (1) feature permutations from the *independent* `FeatureShuffler` stream;
/// then, from the main stream: (2) class-shift base offsets, (3) categorical permutations
/// per-member (only if enabled), (4) row-subsample patterns per-member (only if enabled),
/// (5) one `shuffle()` over the zipped per-member tuples; norm-method assignment happens last,
/// by position, consuming no RNG.
pub fn generate_ensemble(p: &EnsembleConfigParams) -> Vec<MemberConfig> {
    let shuffle_patterns = generate_feature_permutations(p.n_features, p.n_estimators, p.random_state);
    let n_members_from_shuffle = shuffle_patterns.len();

    let mut rng = PyRandom::new(p.random_state);

    let shift_offsets: Vec<usize> = if p.is_classification && p.class_shift && p.n_estimators > 1 && p.n_classes > 1 {
        let base_offsets = rng.sample_indices(p.n_classes, p.n_classes);
        let num_cycles = p.n_estimators.div_ceil(base_offsets.len());
        base_offsets
            .iter()
            .cycle()
            .take(base_offsets.len() * num_cycles)
            .take(p.n_estimators)
            .copied()
            .collect()
    } else {
        vec![0usize; p.n_estimators]
    };

    let cat_permutations: Vec<Option<Vec<Vec<usize>>>> = if p.permute_categorical && !p.cat_value_counts.is_empty() {
        (0..p.n_estimators)
            .map(|_| {
                Some(
                    p.cat_value_counts
                        .iter()
                        .map(|&n_vals| rng.sample_indices(n_vals, n_vals))
                        .collect(),
                )
            })
            .collect()
    } else {
        vec![None; p.n_estimators]
    };

    let n_rows_target = p.max_num_rows.map(|m| m.min(p.n_train_rows)).unwrap_or(p.n_train_rows);
    let row_subsample_patterns: Vec<Option<Vec<usize>>> = if n_rows_target < p.n_train_rows {
        (0..p.n_estimators).map(|_| Some(rng.sample_indices(p.n_train_rows, n_rows_target))).collect()
    } else {
        vec![None; p.n_estimators]
    };

    let n = n_members_from_shuffle.min(shift_offsets.len());
    let mut zipped: Vec<(Vec<usize>, usize, Option<Vec<Vec<usize>>>, Option<Vec<usize>>)> = (0..n)
        .map(|i| {
            (
                shuffle_patterns[i].clone(),
                shift_offsets[i],
                cat_permutations[i].clone(),
                row_subsample_patterns[i].clone(),
            )
        })
        .collect();
    rng.shuffle(&mut zipped);

    let num_cycles = p.n_estimators.div_ceil(p.norm_methods.len());
    let norm_methods_for_estimators: Vec<NormMethod> =
        p.norm_methods.iter().cycle().take(p.norm_methods.len() * num_cycles).take(n).copied().collect();

    zipped
        .into_iter()
        .zip(norm_methods_for_estimators)
        .map(|((feature_permutation, class_shift, cat_permutation, row_subsample), norm_method)| MemberConfig {
            feature_permutation,
            class_shift,
            cat_permutation,
            row_subsample,
            norm_method,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_shape() {
        let p = EnsembleConfigParams {
            n_estimators: 32,
            n_features: 8,
            n_train_rows: 10,
            is_classification: true,
            n_classes: 3,
            class_shift: true,
            permute_categorical: false,
            cat_value_counts: vec![],
            max_num_rows: None,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            random_state: 42,
        };
        let configs = generate_ensemble(&p);
        assert_eq!(configs.len(), 32);
        for (i, c) in configs.iter().enumerate() {
            assert_eq!(c.feature_permutation.len(), 8);
            let mut sorted = c.feature_permutation.clone();
            sorted.sort();
            assert_eq!(sorted, (0..8).collect::<Vec<_>>(), "member {i} not a valid permutation");
            assert!(c.class_shift < 3);
            assert!(c.cat_permutation.is_none());
            assert!(c.row_subsample.is_none());
        }
        // norm methods cycle none/power by final position, deterministically.
        assert_eq!(configs[0].norm_method, NormMethod::None);
        assert_eq!(configs[1].norm_method, NormMethod::Power);
    }

    #[test]
    fn test_small_feature_count_uses_permutation_enumeration() {
        let p = EnsembleConfigParams {
            n_estimators: 32,
            n_features: 3,
            n_train_rows: 10,
            is_classification: false,
            n_classes: 1,
            class_shift: true,
            permute_categorical: false,
            cat_value_counts: vec![],
            max_num_rows: None,
            norm_methods: vec![NormMethod::None],
            random_state: 42,
        };
        let configs = generate_ensemble(&p);
        // 3! = 6 total permutations, fewer than n_estimators=32 -> truncated ensemble.
        assert_eq!(configs.len(), 6);
    }

    #[test]
    fn test_matches_real_ensemble_generator() {
        // Generated by directly invoking the real `EnsembleGenerator(n_estimators=32,
        // norm_methods=None, class_shift=True, random_state=42, task="classification").fit(X, y)`
        // (8 features, 10 rows, 3 classes) from classifier_and_regressor.py and reading back
        // `ensemble_configs_` (grouped by norm method: "none" gets even overall positions,
        // "power" gets odd, per the `norm_methods_for_estimators[i % 2]` cycling below).
        let p = EnsembleConfigParams {
            n_estimators: 32,
            n_features: 8,
            n_train_rows: 10,
            is_classification: true,
            n_classes: 3,
            class_shift: true,
            permute_categorical: false,
            cat_value_counts: vec![],
            max_num_rows: None,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            random_state: 42,
        };
        let configs = generate_ensemble(&p);
        let expected: [(NormMethod, &[usize], usize); 6] = [
            (NormMethod::None, &[0, 5, 3, 4, 7, 1, 6, 2], 1),
            (NormMethod::Power, &[3, 0, 1, 4, 6, 7, 5, 2], 0),
            (NormMethod::None, &[2, 1, 5, 3, 6, 4, 7, 0], 1),
            (NormMethod::Power, &[4, 1, 6, 7, 2, 3, 5, 0], 2),
            (NormMethod::None, &[1, 4, 7, 2, 3, 0, 6, 5], 2),
            (NormMethod::Power, &[2, 3, 1, 7, 6, 0, 4, 5], 0),
        ];
        for (i, (exp_method, exp_perm, exp_shift)) in expected.iter().enumerate() {
            assert_eq!(configs[i].norm_method, *exp_method, "member {i} norm_method");
            assert_eq!(&configs[i].feature_permutation, exp_perm, "member {i} permutation");
            assert_eq!(configs[i].class_shift, *exp_shift, "member {i} shift");
        }
    }
}
