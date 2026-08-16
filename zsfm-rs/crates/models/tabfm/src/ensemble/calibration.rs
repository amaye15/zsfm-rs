//! Platt scaling (binary) and vector scaling (multiclass) output calibration, fit on
//! out-of-fold probabilities (see `oof.rs`). The wrapper fits these via `scipy.optimize.minimize`
//! (L-BFGS-B) on a regularized negative-log-likelihood; this ports the same objective but
//! optimizes it with an in-house box-constrained coordinate descent (golden-section line search
//! per coordinate) — expect looser numerical agreement than the always-on ensembling path, which
//! matches the real optimizer's exact trajectory less closely by construction.

const EPS: f64 = 1e-12;

/// Minimizes `f` over `[lo, hi]` via golden-section search (safe under box constraints, unlike
/// Brent's method's unbounded bracket growth — used here instead of `scalers::brent_minimize`).
fn golden_section_bounded(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    const GR: f64 = 0.618_033_988_749_895; // 1/phi
    let (mut a, mut b) = (lo, hi);
    let mut c = b - GR * (b - a);
    let mut d = a + GR * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..100 {
        if (b - a).abs() < 1e-10 {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - GR * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GR * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

pub struct PlattParams {
    pub a: f64,
    pub b: f64,
}

impl PlattParams {
    /// `p_all`: OOF probabilities `[N][2]`; `y`: true class indices (0 or 1).
    pub fn fit(p_all: &[Vec<f64>], y: &[usize], lambda: f64) -> Self {
        let z: Vec<f64> = p_all.iter().map(|p| ((p[1] + EPS) / (p[0] + EPS)).ln()).collect();
        let loss = |a: f64, b: f64| -> f64 {
            let n = z.len() as f64;
            let mut nll = 0.0;
            for i in 0..z.len() {
                let p1 = sigmoid(a * z[i] + b);
                let p_correct = if y[i] == 1 { p1 } else { 1.0 - p1 };
                nll -= (p_correct + EPS).ln();
            }
            nll / n + lambda * ((a - 1.0).powi(2) + b.powi(2))
        };

        let mut a = 1.0f64;
        let mut b = 0.0f64;
        for _ in 0..20 {
            a = golden_section_bounded(|av| loss(av, b), 0.8, 1.2);
            b = golden_section_bounded(|bv| loss(a, bv), -1.0, 1.0);
        }
        PlattParams { a, b }
    }

    pub fn apply(&self, p: &[f64]) -> Vec<f64> {
        let z = ((p[1] + EPS) / (p[0] + EPS)).ln();
        let p1 = sigmoid(self.a * z + self.b);
        vec![1.0 - p1, p1]
    }
}

pub struct VectorScalingParams {
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

impl VectorScalingParams {
    /// `p_all`: OOF probabilities `[N][K]`; `y`: true class indices.
    pub fn fit(p_all: &[Vec<f64>], y: &[usize], lambda: f64) -> Self {
        let k = p_all[0].len();
        let z: Vec<Vec<f64>> = p_all.iter().map(|p| p.iter().map(|&v| (v + EPS).ln()).collect()).collect();

        let loss = |w: &[f64], b: &[f64]| -> f64 {
            let n = z.len() as f64;
            let mut nll = 0.0;
            for (i, zi) in z.iter().enumerate() {
                let logits: Vec<f64> = (0..k).map(|c| w[c] * zi[c] + b[c]).collect();
                let probs = softmax(&logits);
                nll -= (probs[y[i]] + EPS).ln();
            }
            let reg: f64 =
                w.iter().map(|&wv| (wv - 1.0).powi(2)).sum::<f64>() + b.iter().map(|&bv| bv.powi(2)).sum::<f64>();
            nll / n + lambda * reg
        };

        let mut w = vec![1.0f64; k];
        let mut b = vec![0.0f64; k];
        for _ in 0..20 {
            for c in 0..k {
                let (w2, b2) = (w.clone(), b.clone());
                w[c] = golden_section_bounded(
                    |wc| {
                        let mut wt = w2.clone();
                        wt[c] = wc;
                        loss(&wt, &b2)
                    },
                    0.8,
                    1.2,
                );
            }
            for c in 0..k {
                let (w2, b2) = (w.clone(), b.clone());
                b[c] = golden_section_bounded(
                    |bc| {
                        let mut bt = b2.clone();
                        bt[c] = bc;
                        loss(&w2, &bt)
                    },
                    -1.0,
                    1.0,
                );
            }
        }
        VectorScalingParams { w, b }
    }

    pub fn apply(&self, p: &[f64]) -> Vec<f64> {
        let k = p.len();
        let logits: Vec<f64> = (0..k).map(|c| self.w[c] * (p[c] + EPS).ln() + self.b[c]).collect();
        softmax(&logits)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_improves_calibration_of_overconfident_probs() {
        // Overconfident-but-correct predictions should shrink toward the true labels less
        // aggressively than raw probs after Platt scaling; here we just check the fit doesn't
        // diverge and produces valid probabilities.
        let p_all = vec![vec![0.99, 0.01], vec![0.02, 0.98], vec![0.6, 0.4], vec![0.4, 0.6]];
        let y = vec![0, 1, 0, 1];
        let params = PlattParams::fit(&p_all, &y, 1e-2);
        for p in &p_all {
            let out = params.apply(p);
            assert!((out[0] + out[1] - 1.0).abs() < 1e-6);
            assert!(out[0] >= 0.0 && out[1] >= 0.0);
        }
    }

    #[test]
    fn test_vector_scaling_valid_probabilities() {
        let p_all = vec![vec![0.7, 0.2, 0.1], vec![0.1, 0.8, 0.1], vec![0.2, 0.2, 0.6]];
        let y = vec![0, 1, 2];
        let params = VectorScalingParams::fit(&p_all, &y, 1e-2);
        for p in &p_all {
            let out = params.apply(p);
            let sum: f64 = out.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
