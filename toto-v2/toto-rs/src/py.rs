use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::infer::{TotoModel, InferConfig};

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
pub struct Toto {
    model: TotoModel,
    patch_size: usize,
    max_ctx: usize,
}

#[pymethods]
impl Toto {
    #[new]
    #[pyo3(signature = (gguf, config, context_length=None))]
    pub fn new(gguf: &str, config: &str, context_length: Option<usize>) -> PyResult<Self> {
        let config_str = std::fs::read_to_string(config)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let toto_config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let infer_config = InferConfig {
            d_model:                         toto_config["d_model"].as_u64().unwrap_or(2048) as usize,
            num_layers:                      toto_config["num_layers"].as_u64().unwrap_or(48) as usize,
            num_heads:                       toto_config["num_heads"].as_u64().unwrap_or(32) as usize,
            num_groups:                      toto_config["num_groups"].as_u64().unwrap_or(32) as usize,
            qk_dim:                          toto_config["qk_dim"].as_u64().unwrap_or(64) as usize,
            v_dim:                           toto_config["v_dim"].as_u64().unwrap_or(64) as usize,
            patch_size:                      toto_config["patch_size"].as_u64().unwrap_or(32) as usize,
            norm_eps:                        toto_config["norm_eps"].as_f64().unwrap_or(5e-4),
            layer_group_size:                toto_config["layer_group_size"].as_u64().unwrap_or(48) as usize,
            num_variate_layers_per_group:    toto_config["num_variate_layers_per_group"].as_u64().unwrap_or(1) as usize,
            variate_layer_first:             toto_config["variate_layer_first"].as_bool().unwrap_or(false),
            use_xpos:                        toto_config["use_xpos"].as_bool().unwrap_or(true),
            residual_mult:                   toto_config["residual_mult"].as_f64().unwrap_or(0.75),
            residual_attn_ratio:             toto_config["residual_attn_ratio"].as_f64().unwrap_or(5.136215466577748),
            compute_f64:                     false,
        };
        let patch_size = infer_config.patch_size;
        let max_ctx = context_length.unwrap_or(4096);
        let model = TotoModel::load(std::path::Path::new(gguf), infer_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, patch_size, max_ctx })
    }

    pub fn forecast(
        &self,
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        horizon: usize,
    ) -> PyResult<Py<PyAny>> {
        let contexts = parse_contexts(context)?;
        let quantile_levels = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let median_idx = 4;
        let mut fc_choices = Vec::new();
        for raw_ctx in &contexts {
            if raw_ctx.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err("context must be non-empty"));
            }
            let total_len = raw_ctx.len();
            let usable = total_len.min(self.max_ctx);
            let ctx_len = (usable / self.patch_size) * self.patch_size;
            if ctx_len == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "need at least {} timesteps, got {}",
                    self.patch_size,
                    raw_ctx.len()
                )));
            }
            let ctx = raw_ctx[total_len.saturating_sub(ctx_len)..].to_vec();
            let mask = vec![vec![true; ctx_len]];
            let data = vec![ctx];
            let qmat = self
                .model
                .forecast(&data, &mask, horizon)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let point = qmat
                .get(median_idx)
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_default();
            let mut quantiles = BTreeMap::new();
            for (qi, &level) in quantile_levels.iter().enumerate() {
                if let Some(variate) = qmat.get(qi) {
                    if let Some(q) = variate.first() {
                        quantiles.insert(format!("{level:.1}"), q.clone());
                    }
                }
            }
            fc_choices.push((point, quantiles));
        }
        let total_ctx: usize = contexts.iter().map(|c| c.len()).sum();
        to_py(py, &build_json("toto", total_ctx, horizon, &fc_choices))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Toto>()?;
    Ok(())
}
