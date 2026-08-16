use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::forecast::QuantileMatrix;

#[derive(Serialize)]
#[serde(untagged)]
pub enum ForecastOutput {
    Univariate {
        point: Vec<f32>,
        quantiles: BTreeMap<String, Vec<f32>>,
    },
    Multivariate {
        variates: Vec<VariateForecast>,
    },
}

#[derive(Serialize)]
pub struct VariateForecast {
    pub point: Vec<f32>,
    pub quantiles: BTreeMap<String, Vec<f32>>,
}

/// Turn one forecast's `QuantileMatrix` into a `ForecastOutput`, choosing the
/// univariate (flat `point`/`quantiles`) or multivariate (`variates: [...]`)
/// shape based on how many variates the matrix actually has.
///
/// `quantile_levels[i]` is the probability level for `qmat[i]`; `median_idx`
/// is the row holding the point forecast (usually the index of level 0.5).
pub fn quantile_matrix_to_output(
    qmat: &QuantileMatrix,
    quantile_levels: &[f32],
    median_idx: usize,
) -> ForecastOutput {
    let n_var = qmat.first().map(|q| q.len()).unwrap_or(0);

    let quantiles_for = |vi: usize| -> BTreeMap<String, Vec<f32>> {
        let mut quantiles = BTreeMap::new();
        for (qi, &level) in quantile_levels.iter().enumerate() {
            if let Some(row) = qmat.get(qi) {
                if let Some(q) = row.get(vi) {
                    quantiles.insert(format!("{level:.2}"), q.clone());
                }
            }
        }
        quantiles
    };
    let point_for = |vi: usize| -> Vec<f32> {
        qmat.get(median_idx).and_then(|v| v.get(vi)).cloned().unwrap_or_default()
    };

    if n_var <= 1 {
        ForecastOutput::Univariate {
            point: point_for(0),
            quantiles: quantiles_for(0),
        }
    } else {
        let variates = (0..n_var)
            .map(|vi| VariateForecast { point: point_for(vi), quantiles: quantiles_for(vi) })
            .collect();
        ForecastOutput::Multivariate { variates }
    }
}

/// Wrap one or more `ForecastOutput`s in the shared response envelope every
/// model's `infer` CLI subcommand (and later, Python binding) returns.
pub fn forecast_response_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    outputs: Vec<ForecastOutput>,
) -> anyhow::Result<String> {
    #[derive(Serialize)]
    struct ForecastResponse {
        id: String,
        object: &'static str,
        created: u64,
        model: String,
        choices: Vec<Choice>,
        usage: Usage,
    }
    #[derive(Serialize)]
    struct Choice {
        index: usize,
        forecast: ForecastOutput,
        finish_reason: &'static str,
    }
    #[derive(Serialize)]
    struct Usage {
        context_length: usize,
        forecast_length: usize,
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let resp = ForecastResponse {
        id: format!("forecast-{created:016x}"),
        object: "forecast",
        created,
        model: model_name.to_string(),
        choices: outputs
            .into_iter()
            .enumerate()
            .map(|(i, forecast)| Choice { index: i, forecast, finish_reason: "stop" })
            .collect(),
        usage: Usage { context_length, forecast_length },
    };

    Ok(serde_json::to_string_pretty(&resp)?)
}
