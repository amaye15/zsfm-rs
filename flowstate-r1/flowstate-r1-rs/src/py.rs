use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::FlowStateConfig;
use crate::infer::{FlowStateModel, InferConfig};

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
pub struct FlowState {
    model: FlowStateModel,
    quantile_levels: Vec<f32>,
    median_idx: usize,
}

#[pymethods]
impl FlowState {
    #[new]
    pub fn new(gguf: &str, config: &str) -> PyResult<Self> {
        let config_str = std::fs::read_to_string(config)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let fs_config = FlowStateConfig::from_json(&config_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let infer_config = InferConfig {
            num_layers:        fs_config.encoder_num_layers as usize,
            embed_dim:         fs_config.embedding_feature_dim as usize,
            state_dim:         fs_config.encoder_state_dim as usize,
            n_inputs:          fs_config.n_inputs() as usize,
            decoder_dim:       fs_config.decoder_dim as usize,
            decoder_patch_len: fs_config.decoder_patch_len as usize,
            quantiles:         fs_config.quantiles.clone(),
            basis_range:       fs_config.basis_range(),
            context_length:    fs_config.context_length as usize,
            eps:               1e-5,
        };
        let quantile_levels = infer_config.quantiles.clone();
        let median_idx = quantile_levels
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(quantile_levels.len() / 2);
        let model = FlowStateModel::load(std::path::Path::new(gguf), infer_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, quantile_levels, median_idx })
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
            if ctx.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err("context must be non-empty"));
            }
            let qmat = self
                .model
                .forecast(ctx, horizon)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let point = qmat.get(self.median_idx).cloned().unwrap_or_default();
            let mut quantiles = BTreeMap::new();
            for (i, &level) in self.quantile_levels.iter().enumerate() {
                if let Some(q) = qmat.get(i) {
                    quantiles.insert(format!("{level:.2}"), q.clone());
                }
            }
            fc_choices.push((point, quantiles));
        }
        let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
        to_py(py, &build_json("flowstate-r1", total_ctx, horizon, &fc_choices))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlowState>()?;
    Ok(())
}
