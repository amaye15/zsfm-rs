//! Per-column feature scalers, ported from `tabfm/src/classifier_and_regressor.py`'s
//! `PreprocessingPipeline`, applied in this exact order: `CustomStandardScaler` -> one of 5
//! optional normalizers (if not `"none"`) -> `OutlierRemover` (last, not second — verified
//! against `PreprocessingPipeline.fit`). Each scaler exposes `fit`/`transform` mirroring
//! sklearn's split so the same fitted state (from training columns) can be applied to held-out
//! columns.

/// `CustomStandardScaler`: `clip((x - mean) / (std + eps), -100, 100)`.
pub struct CustomStandardScaler {
    mean: f64,
    scale: f64,
}

impl CustomStandardScaler {
    const EPSILON: f64 = 1e-6;
    const CLIP: f64 = 100.0;

    pub fn fit(x: &[f64]) -> Self {
        let mean = mean(x);
        let scale = std_dev(x, mean, 0) + Self::EPSILON;
        CustomStandardScaler { mean, scale }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&v| ((v - self.mean) / self.scale).clamp(-Self::CLIP, Self::CLIP))
            .collect()
    }
}

/// `OutlierRemover` (`threshold=4.0` default): two-pass mean/std (outliers masked before the
/// second pass), then a smooth log-based soft clip (NOT a hard clip) at the recomputed bounds.
pub struct OutlierRemover {
    lower_bound: f64,
    upper_bound: f64,
}

impl OutlierRemover {
    pub fn fit(x: &[f64], threshold: f64) -> Self {
        let mean1 = mean(x);
        let std1 = std_dev(x, mean1, 1).max(1e-6);
        let lower1 = mean1 - threshold * std1;
        let upper1 = mean1 + threshold * std1;

        let clean: Vec<f64> = x
            .iter()
            .copied()
            .filter(|&v| v >= lower1 && v <= upper1)
            .collect();
        let (mean2, std2) = if clean.is_empty() {
            (mean1, std1)
        } else {
            let m = mean(&clean);
            (m, std_dev(&clean, m, 1).max(1e-6))
        };

        OutlierRemover {
            lower_bound: mean2 - threshold * std2,
            upper_bound: mean2 + threshold * std2,
        }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&v| {
                let v = (-((v.abs()).ln_1p()) + self.lower_bound).max(v);
                (v.abs().ln_1p() + self.upper_bound).min(v)
            })
            .collect()
    }
}

/// Plain `sklearn.preprocessing.StandardScaler` (ddof=0, no epsilon/clip) — used once, globally,
/// on the raw regression target (`y_scaler_` in `TabFMRegressor`), separate from the per-member
/// `CustomStandardScaler` applied to features.
pub struct StandardScaler {
    mean: f64,
    scale: f64,
}

impl StandardScaler {
    pub fn fit(x: &[f64]) -> Self {
        let mean = mean(x);
        let std = std_dev(x, mean, 0);
        StandardScaler { mean, scale: if std == 0.0 { 1.0 } else { std } }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| (v - self.mean) / self.scale).collect()
    }

    pub fn inverse_transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v * self.scale + self.mean).collect()
    }

    pub fn inverse_transform_scalar(&self, v: f64) -> f64 {
        v * self.scale + self.mean
    }
}

/// `RobustScaler(unit_variance=True)`: `(x - median) / (IQR / 1.349...)`, where the divisor
/// makes the scale consistent with a standard-normal's std (`1.349... = Φ⁻¹(0.75) - Φ⁻¹(0.25)`).
pub struct RobustScaler {
    median: f64,
    scale: f64,
}

impl RobustScaler {
    pub fn fit(x: &[f64]) -> Self {
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile_linear(&sorted, 50.0);
        let q25 = percentile_linear(&sorted, 25.0);
        let q75 = percentile_linear(&sorted, 75.0);
        let iqr = q75 - q25;
        let norm_factor = norm_ppf(0.75) - norm_ppf(0.25); // ~1.3489795...
        let scale = if iqr == 0.0 { 1.0 } else { iqr / norm_factor };
        RobustScaler { median, scale }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| (v - self.median) / self.scale).collect()
    }
}

/// `QuantileTransformer(output_distribution="normal")`: empirical CDF (via linear-interpolated
/// percentiles, sklearn's default `n_quantiles=1000` capped at the sample count) mapped through
/// the inverse standard-normal CDF.
pub struct QuantileTransformer {
    references: Vec<f64>, // uniform grid in [0,1], length n_quantiles
    quantiles: Vec<f64>,  // data values at each reference quantile, monotonic non-decreasing
}

impl QuantileTransformer {
    const BOUNDS_THRESHOLD: f64 = 1e-7;

    pub fn fit(x: &[f64]) -> Self {
        let n_quantiles = 1000usize.min(x.len().max(1));
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let references: Vec<f64> = (0..n_quantiles)
            .map(|i| i as f64 / (n_quantiles - 1).max(1) as f64)
            .collect();
        let mut quantiles: Vec<f64> =
            references.iter().map(|&r| percentile_linear(&sorted, r * 100.0)).collect();
        // Enforce monotonicity (sklearn does this to guard against numerical noise).
        for i in 1..quantiles.len() {
            if quantiles[i] < quantiles[i - 1] {
                quantiles[i] = quantiles[i - 1];
            }
        }
        QuantileTransformer { references, quantiles }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&v| {
                let u = interp_monotonic(v, &self.quantiles, &self.references)
                    .clamp(Self::BOUNDS_THRESHOLD, 1.0 - Self::BOUNDS_THRESHOLD);
                norm_ppf(u)
            })
            .collect()
    }
}

/// `PowerTransformer(method="yeo-johnson", standardize=True)`: per-feature MLE-fit `lambda`
/// (via a ported `scipy.optimize.brent`), then standardize the transformed values.
pub struct PowerTransformer {
    lambda: f64,
    post_mean: f64,
    post_std: f64,
}

impl PowerTransformer {
    pub fn fit(x: &[f64]) -> Self {
        let lambda = yeo_johnson_optimize(x);
        let transformed: Vec<f64> = x.iter().map(|&v| yeo_johnson_transform(v, lambda)).collect();
        let post_mean = mean(&transformed);
        let post_std = std_dev(&transformed, post_mean, 0).max(1e-300);
        PowerTransformer { lambda, post_mean, post_std }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&v| (yeo_johnson_transform(v, self.lambda) - self.post_mean) / self.post_std)
            .collect()
    }
}

/// One member's full `PreprocessingPipeline` for a single column: fit on `train_col`, apply to
/// both `train_col` and `test_col`. Order: `CustomStandardScaler` -> normalizer (if not
/// `NormMethod::None`) -> `OutlierRemover`.
pub fn apply_pipeline(
    train_col: &[f64],
    test_col: &[f64],
    norm_method: super::config_gen::NormMethod,
    outlier_threshold: f64,
) -> (Vec<f64>, Vec<f64>) {
    use super::config_gen::NormMethod;

    let scaler = CustomStandardScaler::fit(train_col);
    let mut train = scaler.transform(train_col);
    let mut test = scaler.transform(test_col);

    match norm_method {
        NormMethod::None => {}
        NormMethod::Power => {
            let n = PowerTransformer::fit(&train);
            train = n.transform(&train);
            test = n.transform(&test);
        }
        NormMethod::Quantile => {
            let n = QuantileTransformer::fit(&train);
            train = n.transform(&train);
            test = n.transform(&test);
        }
        NormMethod::QuantileRtdl => {
            // Noise injection uses our own seeded RNG (not bit-identical to NumPy's
            // default_rng/PCG64) — a documented gap, see the plan's scoping note; the
            // downstream QuantileTransformer + StandardScaler math is otherwise exact.
            let noisy = inject_rtdl_noise(&train);
            let n = QuantileTransformer::fit(&noisy);
            let train_q = n.transform(&noisy);
            let test_q = n.transform(&test);
            let std = StandardScaler::fit(&train_q);
            train = std.transform(&train_q);
            test = std.transform(&test_q);
        }
        NormMethod::Robust => {
            let n = RobustScaler::fit(&train);
            train = n.transform(&train);
            test = n.transform(&test);
        }
    }

    let outlier = OutlierRemover::fit(&train, outlier_threshold);
    (outlier.transform(&train), outlier.transform(&test))
}

/// `RTDLQuantileTransformer`'s noise-injection step (`noise=1e-3` default): `x + (noise /
/// max(std(x), noise)) * standard_normal(shape)`. Uses a simple seeded LCG-derived Gaussian
/// (Box-Muller), not NumPy's PCG64 — see `apply_pipeline`'s doc comment.
fn inject_rtdl_noise(x: &[f64]) -> Vec<f64> {
    let std = std_dev(x, mean(x), 0);
    let noise = 1e-3;
    let noise_std = noise / std.max(noise);
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next_u64 = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    x.iter()
        .map(|&v| {
            let u1 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let z = (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            v + noise_std * z
        })
        .collect()
}

fn yeo_johnson_transform(x: f64, lambda: f64) -> f64 {
    const EPS: f64 = 1e-6;
    if x >= 0.0 {
        if lambda.abs() > EPS {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        } else {
            (x + 1.0).ln()
        }
    } else if (lambda - 2.0).abs() > EPS {
        -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
    } else {
        -(-x + 1.0).ln()
    }
}

/// Negative log-likelihood of the Yeo-Johnson transform at `lambda`, matching sklearn's
/// `PowerTransformer._yeo_johnson_optimize`'s objective (maximized there; minimized here).
fn yeo_johnson_neg_log_likelihood(x: &[f64], lambda: f64) -> f64 {
    let n = x.len() as f64;
    let transformed: Vec<f64> = x.iter().map(|&v| yeo_johnson_transform(v, lambda)).collect();
    let m = mean(&transformed);
    let var = variance(&transformed, m, 0);
    if var < f64::MIN_POSITIVE {
        return f64::INFINITY;
    }
    let mut loglike = -n / 2.0 * var.ln();
    let sum_term: f64 = x.iter().map(|&v| v.signum() * v.abs().ln_1p()).sum();
    loglike += (lambda - 1.0) * sum_term;
    -loglike
}

fn yeo_johnson_optimize(x: &[f64]) -> f64 {
    brent_minimize(|lambda| yeo_johnson_neg_log_likelihood(x, lambda), -2.0, 2.0)
}

/// A 1-D bracket-then-Brent minimizer, matching the shape of `scipy.optimize.brent`: first grows
/// an `(a, b, c)` triplet bracketing a minimum from the two starting points (golden-ratio
/// expansion), then refines with Brent's method (golden section + parabolic interpolation).
fn brent_minimize(f: impl Fn(f64) -> f64, xa0: f64, xb0: f64) -> f64 {
    const GOLD: f64 = 1.618_034;
    const GLIMIT: f64 = 100.0;
    const TINY: f64 = 1e-21;

    let (mut ax, mut bx) = (xa0, xb0);
    let (mut fa, mut fb) = (f(ax), f(bx));
    if fa < fb {
        std::mem::swap(&mut ax, &mut bx);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut cx = bx + GOLD * (bx - ax);
    let mut fc = f(cx);

    while fc < fb {
        let r = (bx - ax) * (fb - fc);
        let q = (bx - cx) * (fb - fa);
        let denom = 2.0 * (q - r).abs().max(TINY) * (q - r).signum();
        let mut u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
        let ulim = bx + GLIMIT * (cx - bx);

        let fu;
        if (bx - u) * (u - cx) > 0.0 {
            fu = f(u);
            if fu < fc {
                ax = bx;
                bx = u;
                fa = fb;
                fb = fu;
                break;
            } else if fu > fb {
                cx = u;
                fc = fu;
                break;
            }
            u = cx + GOLD * (cx - bx);
            let fu2 = f(u);
            ax = bx; bx = cx; cx = u;
            fa = fb; fb = fc; fc = fu2;
        } else if (cx - u) * (u - ulim) > 0.0 {
            fu = f(u);
            if fu < fc {
                bx = cx; cx = u; let u2 = cx + GOLD * (cx - bx);
                fb = fc; fc = fu; let fu2 = f(u2);
                ax = bx; bx = cx; cx = u2; fa = fb; fb = fc; fc = fu2;
            } else {
                ax = bx; bx = cx; cx = u;
                fa = fb; fb = fc; fc = fu;
            }
        } else if (u - ulim) * (ulim - cx) >= 0.0 {
            u = ulim;
            fu = f(u);
            ax = bx; bx = cx; cx = u;
            fa = fb; fb = fc; fc = fu;
        } else {
            u = cx + GOLD * (cx - bx);
            fu = f(u);
            ax = bx; bx = cx; cx = u;
            fa = fb; fb = fc; fc = fu;
        }
        if (cx - ax).abs() > 1e6 {
            break; // safety valve against runaway brackets on pathological inputs
        }
    }

    // Ensure ax < cx for Brent's method proper.
    let (mut a, mut c) = if ax < cx { (ax, cx) } else { (cx, ax) };
    let mut x = bx;
    let mut w = bx;
    let mut v = bx;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    const CGOLD: f64 = 0.381_966;
    const ZEPS: f64 = 1e-12;
    const TOL: f64 = 1e-8;

    for _ in 0..100 {
        let xm = 0.5 * (a + c);
        let tol1 = TOL * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (c - a) {
            break;
        }
        let mut use_golden = true;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (c - x) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || c - u < tol2 {
                    d = if xm - x >= 0.0 { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            e = if x >= xm { a - x } else { c - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 { x + d } else { x + if d >= 0.0 { tol1 } else { -tol1 } };
        let fu = f(u);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                c = x;
            }
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                c = u;
            }
            if fu <= fw || w == x {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u; fv = fu;
            }
        }
    }
    x
}

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() { 0.0 } else { x.iter().sum::<f64>() / x.len() as f64 }
}

fn variance(x: &[f64], mean: f64, ddof: usize) -> f64 {
    let n = x.len();
    if n <= ddof {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();
    sum_sq / (n - ddof) as f64
}

fn std_dev(x: &[f64], mean: f64, ddof: usize) -> f64 {
    variance(x, mean, ddof).sqrt()
}

/// NumPy's default ("linear") percentile interpolation on an already-sorted slice.
pub fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// `np.interp(x, xp, fp)`-equivalent: linear interpolation with clamping outside the domain,
/// for a monotonic non-decreasing `xp`.
fn interp_monotonic(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    // Binary search for the interval containing x.
    let idx = xp.partition_point(|&v| v <= x);
    let (x0, x1) = (xp[idx - 1], xp[idx]);
    let (y0, y1) = (fp[idx - 1], fp[idx]);
    if x1 == x0 {
        y0
    } else {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

/// Inverse standard-normal CDF (`scipy.stats.norm.ppf`), via Acklam's rational approximation
/// with one Halley's-method refinement step (accurate to ~1e-9).
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01, 2.209_460_984_245_205e+02, -2.759_285_104_469_687e+02,
        1.383_577_518_672_690e+02, -3.066_479_806_614_716e+01, 2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01, 1.615_858_368_580_409e+02, -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01, -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03, -3.223_964_580_411_365e-01, -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00, 4.374_664_141_464_968e+00, 2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03, 3.224_671_290_700_398e-01, 2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    // One Halley's-method refinement using the standard normal CDF/PDF.
    let e = 0.5 * erfc(-x / std::f64::consts::SQRT_2) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Abramowitz-Stegun 7.1.26 rational approximation to `erf` (max error ~1.5e-7) — sufficient
/// precision for one Halley refinement step above.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_standard_scaler() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = CustomStandardScaler::fit(&x);
        let t = s.transform(&x);
        // mean=3, std (ddof=0) = sqrt(2) ~ 1.41421356
        assert!((t[0] - (1.0 - 3.0) / (2f64.sqrt() + 1e-6)).abs() < 1e-9);
        assert!((t[2]).abs() < 1e-9); // middle value maps to ~0
    }

    #[test]
    fn test_robust_scaler_matches_scipy_norm_factor() {
        let x: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let s = RobustScaler::fit(&x);
        // median of 0..100 is 50; q25=25, q75=75, iqr=50
        assert!((s.median - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_norm_ppf_known_values() {
        assert!((norm_ppf(0.5)).abs() < 1e-6);
        assert!((norm_ppf(0.975) - 1.959_963_985).abs() < 1e-4);
        assert!((norm_ppf(0.025) + 1.959_963_985).abs() < 1e-4);
    }

    #[test]
    fn test_yeo_johnson_identity_at_lambda_1() {
        assert!((yeo_johnson_transform(5.0, 1.0) - 5.0).abs() < 1e-9);
        assert!((yeo_johnson_transform(-3.0, 1.0) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_power_transformer_matches_sklearn() {
        // Generated via: sklearn.preprocessing.PowerTransformer(method="yeo-johnson",
        // standardize=True) fit on np.random.default_rng(0).uniform(-2, 2, 20).
        let x = vec![
            0.5478467492858172, -0.9208531449445188, -1.8361059042552212, -1.9338894578858836,
            1.2530809568010897, 1.6510223091108869, 0.42654310306871945, 0.9179862439359936,
            0.17449996586169148, 1.740289695151073, 1.2634142164861286, -1.9890459993194076,
            1.4296171063502774, -1.8656576987781426, 0.9186217857197763, -1.297377517589764,
            1.4527156893995463, 0.16584488099636685, -0.8011524378504609, -0.3092511152093662,
        ];
        let expected_lambda = 1.345222766603304;
        let expected_transformed = [
            0.2719479783598148, -0.8318522439585222, -1.3650247006836644, -1.4181615227275621,
            0.9606112117567647, 1.3853857310641091, 0.16297351213481462, 0.6223628894658091,
            -0.05314385725391679, 1.483863071047962, 0.9713347980371447, -1.4478643497510573,
            1.1461058305187737, -1.3811492793940139, 0.6229863176690302, -1.0599639970952783,
            1.1707302052556954, -0.06030208166470946, -0.7561721820272899, -0.42466733075390517,
        ];
        let pt = PowerTransformer::fit(&x);
        assert!(
            (pt.lambda - expected_lambda).abs() < 1e-4,
            "lambda {} vs expected {}",
            pt.lambda,
            expected_lambda
        );
        let got = pt.transform(&x);
        for (g, e) in got.iter().zip(expected_transformed.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g} vs expected {e}");
        }
    }

    #[test]
    fn test_percentile_linear() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_linear(&sorted, 50.0) - 3.0).abs() < 1e-9);
        assert!((percentile_linear(&sorted, 0.0) - 1.0).abs() < 1e-9);
        assert!((percentile_linear(&sorted, 100.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_quantile_transformer_matches_sklearn() {
        // Generated via: sklearn.preprocessing.QuantileTransformer(output_distribution="normal",
        // random_state=0) fit on np.random.default_rng(0).uniform(-2, 2, 20). n_quantiles is
        // capped to n_samples=20 by sklearn since 1000 > 20.
        let x = vec![
            0.5478467492858172, -0.9208531449445188, -1.8361059042552212, -1.9338894578858836,
            1.2530809568010897, 1.6510223091108869, 0.42654310306871945, 0.9179862439359936,
            0.17449996586169148, 1.740289695151073, 1.2634142164861286, -1.9890459993194076,
            1.4296171063502774, -1.8656576987781426, 0.9186217857197763, -1.297377517589764,
            1.4527156893995463, 0.16584488099636685, -0.8011524378504609, -0.3092511152093662,
        ];
        let expected = [
            0.199201324789267, -0.6336400007797011, -1.003147967662534, -1.6198562586382699,
            0.633640000779701, 1.6198562586382697, 0.0660118123758406, 0.33603814037182306,
            -0.06601181237584074, 5.19933758270342, 0.8045963803603002, -5.199337582605575,
            1.0031479676625337, -1.2521195202652193, 0.47950565333094985, -0.8045963803603002,
            1.2521195202652189, -0.199201324789267, -0.47950565333095013, -0.33603814037182317,
        ];
        let qt = QuantileTransformer::fit(&x);
        let got = qt.transform(&x);
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g} vs expected {e}");
        }
    }
}
