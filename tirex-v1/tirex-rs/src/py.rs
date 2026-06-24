use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::TiRexConfig;
use crate::infer::TiRexModel;

fn to_py(py: Python<'_>, json_str: &str) -> PyResult<Py<PyAny>> {
    let json_mod = py.import("json")?;
    Ok(json_mod.call_method1("loads", (json_str,))?.unbind())
}

fn build_json(
    model_name: &str,
    context_length: usize,
    forecast_length: usize,
    point: &[f32],
    quantiles: &BTreeMap<String, Vec<f32>>,
) -> String {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    serde_json::to_string_pretty(&serde_json::json!({
        "id": format!("forecast-{created:016x}"),
        "object": "forecast",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "forecast": { "point": point, "quantiles": quantiles },
            "finish_reason": "stop"
        }],
        "usage": { "context_length": context_length, "forecast_length": forecast_length }
    }))
    .unwrap()
}

#[pyclass]
pub struct TiRex {
    model: TiRexModel,
    config: TiRexConfig,
}

#[pymethods]
impl TiRex {
    #[new]
    pub fn new(gguf: &str) -> PyResult<Self> {
        let config = TiRexConfig::default_from_ckpt();
        let model = TiRexModel::load(std::path::Path::new(gguf), config.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, config })
    }

    /// Forecast from context values.
    ///
    /// Parameters
    /// ----------
    /// context : list[float] | list[list[float]]
    ///     Input time series (single or batch).
    /// horizon : int
    ///     Number of future steps.
    /// all_outputs : bool, optional
    ///     If True, include all quantile forecasts in the response.
    pub fn forecast(
        &self,
        py: Python<'_>,
        context: &Bound<'_, PyAny>,
        horizon: usize,
        all_outputs: bool,
    ) -> PyResult<Py<PyAny>> {
        // Accept single list or list of lists
        let ctx: Vec<f32> = if let Ok(v) = context.extract::<Vec<f32>>() {
            v
        } else {
            context.extract::<Vec<Vec<f32>>>()?.into_iter().flatten().collect()
        };

        if ctx.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("context must be non-empty"));
        }

        let (quantiles, mean) = self.model
            .forecast(&ctx, horizon)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut q_map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        if all_outputs {
            for (i, q) in self.config.quantiles.iter().enumerate() {
                q_map.insert(format!("q{:.1}", q), quantiles[i].clone());
            }
        }

        to_py(py, &build_json("tirex", ctx.len(), horizon, &mean, &q_map))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TiRex>()?;
    Ok(())
}
