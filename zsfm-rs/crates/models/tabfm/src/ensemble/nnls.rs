//! Lawson-Hanson active-set NNLS (`scipy.optimize.nnls`'s algorithm): solves
//! `min ||Ax - b||^2` subject to `x >= 0`. Self-contained (no external linear-algebra crate) —
//! columns are few (`n_estimators <= 32`), so dense Gaussian elimination on the small
//! passive-set normal-equations system is more than adequate.

/// `a` is column-major: `a[j]` is the `j`-th column (length `m`, matching `b`'s length).
/// Returns the `n = a.len()`-length non-negative solution.
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    let m = b.len();
    let mut x = vec![0.0f64; n];
    let mut passive: Vec<usize> = Vec::new();
    let mut active: Vec<usize> = (0..n).collect();

    const TOL: f64 = 1e-10;
    let max_iter = 3 * n + 10;

    for _ in 0..max_iter {
        // w = A^T (b - A x)
        let residual: Vec<f64> = {
            let ax = matvec(a, &x, m);
            (0..m).map(|i| b[i] - ax[i]).collect()
        };
        let w: Vec<f64> = active.iter().map(|&j| dot(&a[j], &residual)).collect();

        let Some((best_pos, &best_w)) = w.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) else {
            break;
        };
        if active.is_empty() || best_w <= TOL {
            break;
        }
        let j = active.remove(best_pos);
        passive.push(j);

        loop {
            let z_passive = solve_normal_equations(&passive.iter().map(|&j| a[j].clone()).collect::<Vec<_>>(), b);
            if z_passive.iter().all(|&v| v > TOL) {
                for (idx, &j) in passive.iter().enumerate() {
                    x[j] = z_passive[idx];
                }
                break;
            }
            // Feasibility step: find alpha shrinking x toward z_passive without crossing 0.
            let mut alpha = f64::INFINITY;
            for (idx, &j) in passive.iter().enumerate() {
                if z_passive[idx] <= TOL {
                    let denom = x[j] - z_passive[idx];
                    if denom > 0.0 {
                        alpha = alpha.min(x[j] / denom);
                    }
                }
            }
            if !alpha.is_finite() {
                alpha = 0.0;
            }
            for (idx, &j) in passive.iter().enumerate() {
                x[j] += alpha * (z_passive[idx] - x[j]);
            }
            // Move near-zero passive indices back to active.
            let mut still_passive = Vec::new();
            for &j in &passive {
                if x[j] <= TOL {
                    x[j] = 0.0;
                    active.push(j);
                } else {
                    still_passive.push(j);
                }
            }
            passive = still_passive;
            if passive.is_empty() {
                break;
            }
        }
    }
    x
}

fn matvec(a: &[Vec<f64>], x: &[f64], m: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m];
    for (j, col) in a.iter().enumerate() {
        if x[j] == 0.0 {
            continue;
        }
        for i in 0..m {
            out[i] += col[i] * x[j];
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Solves `argmin_z ||A_cols z - b||^2` (unconstrained) via the normal equations
/// `(A^T A) z = A^T b`, Gaussian elimination with partial pivoting.
fn solve_normal_equations(a_cols: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let k = a_cols.len();
    if k == 0 {
        return vec![];
    }
    let mut ata = vec![vec![0.0f64; k]; k];
    let mut atb = vec![0.0f64; k];
    for i in 0..k {
        atb[i] = dot(&a_cols[i], b);
        for j in 0..k {
            ata[i][j] = dot(&a_cols[i], &a_cols[j]);
        }
        ata[i][i] += 1e-10; // ridge for numerical stability on near-singular passive sets
    }
    gaussian_solve(&mut ata, &mut atb)
}

fn gaussian_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > pivot_val {
                pivot_val = a[row][col].abs();
                pivot_row = row;
            }
        }
        if pivot_val < 1e-14 {
            continue; // singular column; leave corresponding solution as 0 later
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);
        let diag = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        if a[i][i].abs() < 1e-14 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    x
}

/// The wrapper's ensemble-weight post-processing: normalize NNLS weights to sum to 1 (fallback
/// to uniform if the sum is 0), then blend with uniform via `nnls_beta`.
pub fn finalize_weights(raw_weights: &[f64], nnls_beta: f64) -> Vec<f64> {
    let n = raw_weights.len();
    let sum: f64 = raw_weights.iter().sum();
    let normalized: Vec<f64> = if sum > 0.0 {
        raw_weights.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / n as f64; n]
    };
    let uniform = 1.0 / n as f64;
    normalized.iter().map(|&w| nnls_beta * w + (1.0 - nnls_beta) * uniform).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nnls_exact_solution() {
        // A = [[1,0],[0,1],[1,1]], b = [1,2,3] -> exact solution x=[1,2]
        let a = vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
        let b = vec![1.0, 2.0, 3.0];
        let x = nnls(&a, &b);
        assert!((x[0] - 1.0).abs() < 1e-6, "x0={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-6, "x1={}", x[1]);
    }

    #[test]
    fn test_nnls_enforces_nonnegativity() {
        // A single column strongly anti-correlated with b should be driven to 0, not negative.
        let a = vec![vec![-1.0, -1.0, -1.0]];
        let b = vec![1.0, 1.0, 1.0];
        let x = nnls(&a, &b);
        assert!(x[0] >= 0.0);
    }

    #[test]
    fn test_nnls_matches_scipy() {
        // Generated via: rng = np.random.default_rng(3); A = rng.uniform(-1,1,(10,4));
        // b = rng.uniform(0,1,10); scipy.optimize.nnls(A, b).
        let a = vec![
            vec![-0.8287016657127513, -0.8117427155192016, 0.46915430281842907, -0.1387439591716444, -0.4315976725024171, -0.9970198329823277, 0.7834221408903144, -0.9393079846750576, 0.3210001348557896, -0.4036738186851505],
            vec![-0.5263789868078006, -0.1337461195270524, -0.7726559601571932, 0.17359714287628147, 0.2970944141596501, 0.9469205495328255, 0.17032587978181613, 0.413930191311247, 0.862927709482709, 0.48351336013866075],
            vec![0.6025489304127938, -0.04189740371833195, -0.21754361900867591, 0.4756755745843204, 0.39243199334031087, -0.4031975539662487, -0.057380669636337256, -0.2515123330430584, -0.5856176638379975, 0.44432961628423495],
            vec![0.16432407212873557, -0.6805221707258429, 0.03348036524272735, 0.9125345096721971, -0.41455850197502575, -0.3720279959313264, 0.5465540192976328, -0.8182945729914843, 0.26018039957068595, -0.5625691508623909],
        ];
        let b = vec![0.8298868742743123, 0.6576522108732432, 0.6827989078603502, 0.820075750170535, 0.42857290429846195, 0.758705461154919, 0.8784801846662539, 0.1023199219220744, 0.8497683374661538, 0.39392733263233515];
        let expected = [0.0, 0.4483466391927005, 0.283691444258291, 0.19757999070784782];
        let x = nnls(&a, &b);
        for (g, e) in x.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "got {g} vs expected {e}");
        }
    }

    #[test]
    fn test_finalize_weights_sums_to_one_when_beta_1() {
        let w = finalize_weights(&[1.0, 3.0], 1.0);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-9);
        assert!((w[0] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_finalize_weights_uniform_when_beta_0() {
        let w = finalize_weights(&[1.0, 3.0], 0.0);
        assert!((w[0] - 0.5).abs() < 1e-9);
        assert!((w[1] - 0.5).abs() < 1e-9);
    }
}
