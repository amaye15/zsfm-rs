use anyhow::Context;

/// Maximum stdin JSON payload accepted by `infer` commands (64 MiB).
pub const MAX_STDIN_BYTES: u64 = 64 << 20;
/// Maximum forecast horizon steps accepted by `infer` commands.
pub const MAX_HORIZON: usize = 4096;
/// Maximum total context values accepted per request (flat count).
pub const MAX_CONTEXT_VALUES: usize = 1_000_000;

/// Parse the `context` field of a forecast request into `[batch][variate][time]`.
///
/// Accepted shapes:
/// - `[t0, t1, ...]` → batch=1, n_var=1 (flat single series)
/// - `[[t0, t1, ...], ...]` → batch=N, n_var=1 (batch of univariate)
/// - `[[[v0_t0, ...], [v1_t0, ...]], ...]` → batch=N, n_var=M (batch of multivariate)
pub fn parse_mv_contexts(val: serde_json::Value) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    let parsed = parse_mv_contexts_inner(val)?;
    validate_contexts(&parsed)?;
    Ok(parsed)
}

fn parse_mv_contexts_inner(val: serde_json::Value) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    match val {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            anyhow::bail!("context must be a non-empty array")
        }
        serde_json::Value::Array(arr) => {
            let first = arr.first().context("context must be a non-empty array")?;
            if !first.is_array() {
                // Flat: [t0, t1, ...] → batch=1, n_var=1
                let ctx = serde_json::from_value::<Vec<f32>>(serde_json::Value::Array(arr))
                    .context("context must be a JSON array of numbers")?;
                anyhow::ensure!(!ctx.is_empty(), "context series must not be empty");
                Ok(vec![vec![ctx]])
            } else if first
                .as_array()
                .and_then(|a| a.first())
                .map(|v| v.is_array())
                .unwrap_or(false)
            {
                // 3D: [batch][variate][time]
                let out: Vec<Vec<Vec<f32>>> = arr
                    .into_iter()
                    .enumerate()
                    .map(|(i, batch_item)| {
                        let variates = serde_json::from_value::<Vec<Vec<f32>>>(batch_item)
                            .with_context(|| {
                                format!("context[{i}] must be an array of variate arrays")
                            })?;
                        anyhow::ensure!(!variates.is_empty(), "context[{i}] must not be empty");
                        for (vi, series) in variates.iter().enumerate() {
                            anyhow::ensure!(
                                !series.is_empty(),
                                "context[{i}][{vi}] must not be empty"
                            );
                        }
                        Ok(variates)
                    })
                    .collect::<anyhow::Result<_>>()?;
                anyhow::ensure!(!out.is_empty(), "context must be a non-empty array");
                Ok(out)
            } else {
                // 2D: [batch][time] → each series is n_var=1
                let out: Vec<Vec<Vec<f32>>> = arr
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let series = serde_json::from_value::<Vec<f32>>(v)
                            .with_context(|| format!("context[{i}] must be an array of numbers"))?;
                        anyhow::ensure!(!series.is_empty(), "context[{i}] must not be empty");
                        Ok(vec![series])
                    })
                    .collect::<anyhow::Result<_>>()?;
                anyhow::ensure!(!out.is_empty(), "context must be a non-empty array");
                Ok(out)
            }
        }
        _ => anyhow::bail!("context must be a JSON array"),
    }
}

/// Reject empty, oversized, or non-finite contexts before inference.
pub fn validate_contexts(contexts: &[Vec<Vec<f32>>]) -> anyhow::Result<()> {
    anyhow::ensure!(!contexts.is_empty(), "context must be a non-empty array");
    let mut total: usize = 0;
    for (bi, batch) in contexts.iter().enumerate() {
        anyhow::ensure!(!batch.is_empty(), "context[{bi}] must not be empty");
        for (vi, series) in batch.iter().enumerate() {
            anyhow::ensure!(!series.is_empty(), "context[{bi}][{vi}] must not be empty");
            total = total.saturating_add(series.len());
            anyhow::ensure!(
                total <= MAX_CONTEXT_VALUES,
                "context too large — got {total} values, max is {MAX_CONTEXT_VALUES}"
            );
            for &v in series {
                anyhow::ensure!(
                    v.is_finite(),
                    "context[{bi}][{vi}] must contain only finite numbers"
                );
            }
        }
    }
    Ok(())
}

/// Parse and bound the `horizon` field of a forecast request.
pub fn parse_horizon(val: &serde_json::Value) -> anyhow::Result<usize> {
    let horizon = val
        .get("horizon")
        .and_then(|v| v.as_u64())
        .context("horizon must be a positive integer")? as usize;
    validate_horizon(horizon)?;
    Ok(horizon)
}

/// Enforce `1 <= horizon <= MAX_HORIZON`.
pub fn validate_horizon(horizon: usize) -> anyhow::Result<()> {
    anyhow::ensure!(horizon >= 1, "horizon must be a positive integer");
    anyhow::ensure!(
        horizon <= MAX_HORIZON,
        "horizon too large — got {horizon}, max is {MAX_HORIZON}"
    );
    Ok(())
}

/// Parse a 2D numeric matrix for tabular `infer` requests.
pub fn parse_matrix(val: &serde_json::Value) -> anyhow::Result<Vec<Vec<f32>>> {
    let mat: Vec<Vec<f32>> =
        serde_json::from_value(val.clone()).context("expected a 2D JSON array")?;
    anyhow::ensure!(!mat.is_empty(), "matrix must be a non-empty array");
    let cols = mat[0].len();
    anyhow::ensure!(cols > 0, "matrix rows must not be empty");
    anyhow::ensure!(
        mat.len() * cols <= MAX_CONTEXT_VALUES,
        "matrix too large — max is {MAX_CONTEXT_VALUES} values"
    );
    for (ri, row) in mat.iter().enumerate() {
        anyhow::ensure!(row.len() == cols, "row {ri} has inconsistent width");
        for &v in row {
            anyhow::ensure!(
                v.is_finite(),
                "matrix row {ri} must contain only finite numbers"
            );
        }
    }
    Ok(mat)
}

/// Numerically stable softmax over host `f32` logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // `max` is finite here: callers validate finiteness, empty case handled above.
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

/// Read stdin with a byte cap so a huge pipe cannot OOM before JSON parsing.
pub fn read_stdin_limited() -> anyhow::Result<String> {
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin()
        .take(MAX_STDIN_BYTES + 1)
        .read_to_string(&mut buf)
        .context("read stdin")?;
    anyhow::ensure!(
        (buf.len() as u64) <= MAX_STDIN_BYTES,
        "stdin input too large — max is {MAX_STDIN_BYTES} bytes"
    );
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_contexts() {
        assert!(parse_mv_contexts(serde_json::json!([])).is_err());
        assert!(parse_mv_contexts(serde_json::json!([[]])).is_err());
        assert!(parse_mv_contexts(serde_json::json!([[[]]])).is_err());
    }

    #[test]
    fn rejects_non_finite_context() {
        let nan = f64::NAN;
        // JSON cannot encode NaN, so inject via f32 bit pattern through Value.
        let v = serde_json::json!([1.0, 2.0]);
        let mut ctx = parse_mv_contexts(v).unwrap();
        ctx[0][0][0] = f32::from_bits(0x7fc0_0000);
        assert!(validate_contexts(&ctx).is_err());
        let _ = nan;
    }

    #[test]
    fn horizon_bounds() {
        assert!(validate_horizon(0).is_err());
        assert!(validate_horizon(1).is_ok());
        assert!(validate_horizon(MAX_HORIZON).is_ok());
        assert!(validate_horizon(MAX_HORIZON + 1).is_err());
    }

    #[test]
    fn matrix_validation() {
        assert!(parse_matrix(&serde_json::json!([])).is_err());
        assert!(parse_matrix(&serde_json::json!([[1.0], [2.0, 3.0]])).is_err());
        assert!(parse_matrix(&serde_json::json!([[1.0, 2.0]])).is_ok());
        assert_eq!(softmax(&[]).len(), 0);
        let p = softmax(&[1.0, 2.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
