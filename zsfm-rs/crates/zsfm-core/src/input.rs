use anyhow::Context;

/// Parse the `context` field of a forecast request into `[batch][variate][time]`.
///
/// Accepted shapes:
/// - `[t0, t1, ...]` → batch=1, n_var=1 (flat single series)
/// - `[[t0, t1, ...], ...]` → batch=N, n_var=1 (batch of univariate)
/// - `[[[v0_t0, ...], [v1_t0, ...]], ...]` → batch=N, n_var=M (batch of multivariate)
pub fn parse_mv_contexts(val: serde_json::Value) -> anyhow::Result<Vec<Vec<Vec<f32>>>> {
    match val {
        serde_json::Value::Array(arr) if arr.is_empty() => {
            anyhow::bail!("context must be a non-empty array")
        }
        serde_json::Value::Array(arr) => {
            let first = arr.first().unwrap();
            if !first.is_array() {
                // Flat: [t0, t1, ...] → batch=1, n_var=1
                let ctx = serde_json::from_value::<Vec<f32>>(serde_json::Value::Array(arr))
                    .context("context must be a JSON array of numbers")?;
                Ok(vec![vec![ctx]])
            } else if first
                .as_array()
                .and_then(|a| a.first())
                .map(|v| v.is_array())
                .unwrap_or(false)
            {
                // 3D: [batch][variate][time]
                arr.into_iter()
                    .enumerate()
                    .map(|(i, batch_item)| {
                        serde_json::from_value::<Vec<Vec<f32>>>(batch_item)
                            .with_context(|| format!("context[{i}] must be an array of variate arrays"))
                    })
                    .collect()
            } else {
                // 2D: [batch][time] → each series is n_var=1
                arr.into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let series = serde_json::from_value::<Vec<f32>>(v)
                            .with_context(|| format!("context[{i}] must be an array of numbers"))?;
                        Ok(vec![series])
                    })
                    .collect()
            }
        }
        _ => anyhow::bail!("context must be a JSON array"),
    }
}
