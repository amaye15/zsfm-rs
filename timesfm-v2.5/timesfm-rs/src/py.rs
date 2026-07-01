use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::infer::TimesFMModel;

fn parse_contexts(context: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(outer) = context.extract::<Vec<Vec<f32>>>() {
        return Ok(outer);
    }
    Ok(vec![context.extract::<Vec<f32>>()?])
}

fn to_py(py: Python<'_>, json_str: &str) -> PyResult<Py<PyAny>> {
    let json_mod = py.import("json")?;
    Ok(json_mod.call_method1("loads", (json_str,))?.unbind())
}

fn build_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    choices: &[(Vec<f32>, BTreeMap<String, Vec<f32>>)],
) -> String {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let choices_json: Vec<_> = choices
        .iter()
        .enumerate()
        .map(|(i, (point, quantiles))| {
            serde_json::json!({
                "index": i,
                "forecast": { "point": point, "quantiles": quantiles },
                "finish_reason": "stop"
            })
        })
        .collect();
    serde_json::to_string_pretty(&serde_json::json!({
        "id": format!("forecast-{created:016x}"),
        "object": "forecast",
        "created": created,
        "model": model_name,
        "choices": choices_json,
        "usage": { "context_length": context_length, "forecast_length": forecast_length }
    }))
    .unwrap()
}

#[pyclass]
pub struct TimesFM {
    model: TimesFMModel,
}

#[pymethods]
impl TimesFM {
    #[new]
    pub fn new(gguf: &str) -> PyResult<Self> {
        let model = TimesFMModel::load(std::path::Path::new(gguf))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model })
    }

    pub fn forecast(
        &self,
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        horizon: usize,
    ) -> PyResult<Py<PyAny>> {
        let contexts = parse_contexts(context)?;
        let quantile_labels = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"];
        let mut fc_choices = Vec::new();
        for ctx in &contexts {
            if ctx.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err("context must be non-empty"));
            }
            let outputs = self
                .model
                .forecast(ctx, horizon)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let point = outputs.first().cloned().unwrap_or_default();
            let mut quantiles = BTreeMap::new();
            for (i, label) in quantile_labels.iter().enumerate() {
                if let Some(q) = outputs.get(i + 1) {
                    quantiles.insert(label.to_string(), q.clone());
                }
            }
            fc_choices.push((point, quantiles));
        }
        let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
        to_py(py, &build_json("timesfm", total_ctx, horizon, &fc_choices))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TimesFM>()?;
    Ok(())
}
