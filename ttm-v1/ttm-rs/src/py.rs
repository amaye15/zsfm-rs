use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::TtmConfig;
use crate::infer::TtmModel;

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
pub struct Ttm {
    model: TtmModel,
    min_ctx: usize,
}

#[pymethods]
impl Ttm {
    #[new]
    pub fn new(gguf: &str, config: &str) -> PyResult<Self> {
        let config_str = std::fs::read_to_string(config)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ttm_config = TtmConfig::from_json(&config_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let min_ctx = ttm_config.patch_length;
        let model = TtmModel::load(std::path::Path::new(gguf), ttm_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, min_ctx })
    }

    pub fn forecast(
        &self,
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        horizon: usize,
    ) -> PyResult<Py<PyAny>> {
        let contexts = parse_contexts(context)?;
        let mut fc_choices = Vec::new();
        for ctx in &contexts {
            if ctx.len() < self.min_ctx {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "need at least {} context values, got {}",
                    self.min_ctx,
                    ctx.len()
                )));
            }
            let raw = self
                .model
                .forecast(ctx)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let point: Vec<f32> = raw.into_iter().take(horizon).collect();
            fc_choices.push((point, BTreeMap::<String, Vec<f32>>::new()));
        }
        let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
        to_py(py, &build_json("ttm", total_ctx, horizon, &fc_choices))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Ttm>()?;
    Ok(())
}
