use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Chronos2Config;
use crate::infer::{ChronosModel, InferConfig};

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
pub struct Chronos {
    model: ChronosModel,
    quantile_levels: Vec<f32>,
    patch_size: usize,
    patch_stride: usize,
    max_ctx: usize,
    median_idx: usize,
}

#[pymethods]
impl Chronos {
    #[new]
    pub fn new(gguf: &str, config: &str) -> PyResult<Self> {
        let config_str = std::fs::read_to_string(config)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let c2 = Chronos2Config::from_json(&config_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let cc = &c2.chronos_config;
        let infer_config = InferConfig {
            d_model:             c2.d_model as usize,
            d_kv:                c2.d_kv as usize,
            d_ff:                c2.d_ff as usize,
            num_layers:          c2.num_layers as usize,
            num_heads:           c2.num_heads as usize,
            layer_norm_eps:      c2.layer_norm_epsilon,
            rope_theta:          c2.rope_theta,
            patch_size:          cc.input_patch_size as usize,
            patch_stride:        cc.input_patch_stride as usize,
            context_length:      cc.context_length as usize,
            quantiles:           cc.quantiles.clone(),
            use_reg_token:       cc.use_reg_token,
            use_arcsinh:         cc.use_arcsinh,
            time_encoding_scale: c2.time_encoding_scale() as usize,
            dense_act_fn:        c2.dense_act_fn().to_string(),
        };
        let patch_size = infer_config.patch_size;
        let patch_stride = infer_config.patch_stride;
        let max_ctx = infer_config.context_length;
        let quantile_levels = infer_config.quantiles.clone();
        let median_idx = quantile_levels
            .iter()
            .position(|&q| (q - 0.5).abs() < 1e-6)
            .unwrap_or(quantile_levels.len() / 2);
        let model = ChronosModel::load(std::path::Path::new(gguf), infer_config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, quantile_levels, patch_size, patch_stride, max_ctx, median_idx })
    }

    pub fn forecast(
        &self,
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        horizon: usize,
    ) -> PyResult<Py<PyAny>> {
        let contexts = parse_contexts(context)?;
        let mut fc_choices = Vec::new();
        for mut ctx in contexts.clone() {
            let total_len = ctx.len();
            let usable = total_len.min(self.max_ctx);
            let ctx_len = (usable / self.patch_stride) * self.patch_stride;
            if ctx_len < self.patch_size {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "need at least {} timesteps, got {}",
                    self.patch_size,
                    ctx.len()
                )));
            }
            ctx = ctx[total_len.saturating_sub(ctx_len)..].to_vec();
            let qmat = self
                .model
                .forecast(&ctx, horizon)
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
        to_py(py, &build_json("chronos", total_ctx, horizon, &fc_choices))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chronos>()?;
    Ok(())
}
