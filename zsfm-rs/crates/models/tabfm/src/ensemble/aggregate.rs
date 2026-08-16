//! Temperature-scaled softmax and the classification/regression ensemble-combination paths from
//! `TabFMClassifier._process_logits` / `TabFMRegressor._combine_predictions`.

/// `TabFMClassifier.softmax` (temperature divides logits *before* the max-subtracted softmax).
pub fn softmax_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logits.iter().map(|&v| v / temperature).collect();
    let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

/// Undoes a member's class-label shift: that member was fed `(y_true + shift) % n_classes` as
/// training labels, so its output logit at position `(c + shift) % n_classes` is the prediction
/// for original class `c`.
pub fn unshift_logits(logits: &[f64], shift: usize, n_classes: usize) -> Vec<f64> {
    (0..n_classes).map(|c| logits[(c + shift) % n_classes]).collect()
}

fn average_vecs(vecs: &[Vec<f64>]) -> Vec<f64> {
    let n = vecs.len() as f64;
    let dim = vecs[0].len();
    let mut out = vec![0.0; dim];
    for v in vecs {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += x / n;
        }
    }
    out
}

fn weighted_average_vecs(vecs: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let dim = vecs[0].len();
    let mut out = vec![0.0; dim];
    for (v, &w) in vecs.iter().zip(weights.iter()) {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += x * w;
        }
    }
    out
}

pub enum ClassAggMode<'a> {
    /// Weighted average of per-member (temperature-softmax'd) probabilities.
    NnlsWeighted(&'a [f64]),
    /// Average logits first, then a single temperature-softmax. The wrapper's default
    /// (`average_logits=True`).
    AverageLogits,
    /// Temperature-softmax each member, then plain-average the probabilities.
    AverageProbs,
}

/// `_process_logits`: `logits_all` is `[n_estimators][n_classes]`, already un-shifted back to
/// original class order. Returns final `[n_classes]` probabilities.
pub fn aggregate_classification(logits_all: &[Vec<f64>], temperature: f64, mode: &ClassAggMode) -> Vec<f64> {
    match mode {
        ClassAggMode::NnlsWeighted(weights) => {
            let probs_all: Vec<Vec<f64>> =
                logits_all.iter().map(|l| softmax_temperature(l, temperature)).collect();
            weighted_average_vecs(&probs_all, weights)
        }
        ClassAggMode::AverageLogits => {
            let avg = average_vecs(logits_all);
            softmax_temperature(&avg, temperature)
        }
        ClassAggMode::AverageProbs => {
            let probs_all: Vec<Vec<f64>> =
                logits_all.iter().map(|l| softmax_temperature(l, temperature)).collect();
            average_vecs(&probs_all)
        }
    }
}

/// `_combine_predictions`, NNLS-off (default) path: average the still-*scaled* per-member
/// predictions first; the caller inverse-transforms the single averaged result afterward.
pub fn average_scaled_predictions(scaled: &[f64]) -> f64 {
    scaled.iter().sum::<f64>() / scaled.len() as f64
}

/// `_combine_predictions`, NNLS-on path: each member's prediction has already been
/// inverse-transformed by the caller; combine via the fitted weights.
pub fn weighted_unscaled_predictions(unscaled: &[f64], weights: &[f64]) -> f64 {
    unscaled.iter().zip(weights.iter()).map(|(p, w)| p * w).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let p = softmax_temperature(&[1.0, 2.0, 3.0], 0.9);
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_unshift_roundtrip() {
        // member fed y'=(y+shift)%3; its logit position for original class c is (c+shift)%3.
        let n_classes = 3;
        let shift = 1;
        // suppose the model's raw (shifted-space) confidence peaks at shifted-position 0,
        // meaning it's confident about shifted-class 0 = original class (0 - shift) mod 3 = 2.
        let shifted_logits = vec![9.0, 0.1, 0.2];
        let unshifted = unshift_logits(&shifted_logits, shift, n_classes);
        // unshifted[c] = shifted_logits[(c+shift)%3]; unshifted[2] = shifted_logits[(2+1)%3=0] = 9.0
        assert_eq!(unshifted[2], 9.0);
    }

    #[test]
    fn test_average_logits_vs_average_probs_differ() {
        let logits_all = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let via_logits = aggregate_classification(&logits_all, 0.9, &ClassAggMode::AverageLogits);
        let via_probs = aggregate_classification(&logits_all, 0.9, &ClassAggMode::AverageProbs);
        // average-logits of [10,0] and [0,10] -> [5,5] -> softmax -> [0.5,0.5]
        assert!((via_logits[0] - 0.5).abs() < 1e-6);
        // average-probs: softmax([10,0]/0.9)~[~1,~0], softmax([0,10]/0.9)~[~0,~1] -> avg ~[0.5,0.5] too here
        // (symmetric case coincides) — use an asymmetric case to actually differentiate:
        let logits_all2 = vec![vec![10.0, 0.0], vec![1.0, 0.0]];
        let l2 = aggregate_classification(&logits_all2, 0.9, &ClassAggMode::AverageLogits);
        let p2 = aggregate_classification(&logits_all2, 0.9, &ClassAggMode::AverageProbs);
        assert!((l2[0] - p2[0]).abs() > 1e-3, "expected the two paths to diverge on asymmetric input");
        let _ = via_probs;
    }
}
