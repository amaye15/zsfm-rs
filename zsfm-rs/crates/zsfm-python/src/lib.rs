use pyo3::prelude::*;
use std::path::PathBuf;

/// Python bindings for zsfm — zero-shot forecasting and tabular foundation models.
///
/// Mirrors the Rust CLI (`zsfm <model> convert / infer / delete`) via `uv + pyo3 + maturin`.
/// Every model exposes a `Model` class (e.g. `TtmModel`, `ChronosModel`) with
/// `forecast`/`predict`, plus top-level `convert`/`delete` helpers that dispatch
/// by model name. See `list_models()` for the full registry.
#[pymodule]
fn zsfm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(list_models, m)?)?;
    m.add_function(wrap_pyfunction!(list_forecasters, m)?)?;
    m.add_function(wrap_pyfunction!(list_tabular, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;

    // forecasters
    m.add_class::<TotoModel>()?;
    m.add_class::<ChronosModel>()?;
    m.add_class::<TimesFmModel>()?;
    m.add_class::<SundialModel>()?;
    m.add_class::<TtmModel>()?;
    m.add_class::<LagLlamaModel>()?;
    m.add_class::<MomentModel>()?;
    m.add_class::<MoiraiModel>()?;
    m.add_class::<Moirai2Model>()?;
    m.add_class::<FlowStateModel>()?;
    m.add_class::<TirexModel>()?;

    // tabular
    m.add_class::<MitraModel>()?;
    m.add_class::<TabDptModel>()?;
    m.add_class::<TabIclModel>()?;
    m.add_class::<TabPfnModel>()?;
    m.add_class::<TabFmModel>()?;
    Ok(())
}

#[pyfunction]
fn list_models() -> Vec<&'static str> {
    vec![
        "toto", "chronos", "timesfm", "sundial", "ttm", "lag_llama", "moment", "moirai",
        "moirai2", "flowstate", "tirex", "mitra", "tabdpt", "tabicl", "tabpfn", "tabfm",
    ]
}
#[pyfunction]
fn list_forecasters() -> Vec<&'static str> {
    vec![
        "toto", "chronos", "timesfm", "sundial", "ttm", "lag_llama", "moment", "moirai",
        "moirai2", "flowstate", "tirex",
    ]
}
#[pyfunction]
fn list_tabular() -> Vec<&'static str> {
    vec!["mitra", "tabdpt", "tabicl", "tabpfn", "tabfm"]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dtype_from_str(s: &str) -> PyResult<zsfm_gguf::GGMLType> {
    match s {
        "f32" => Ok(zsfm_gguf::GGMLType::F32),
        "f16" => Ok(zsfm_gguf::GGMLType::F16),
        "q8" => Ok(zsfm_gguf::GGMLType::Q8_0),
        "bf16" => Ok(zsfm_gguf::GGMLType::BF16),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown dtype {s:?}: expected one of f32, f16, q8, bf16"
        ))),
    }
}
fn delete_cached_model(canonical: &std::path::Path, output: Option<&std::path::Path>) -> anyhow::Result<()> {
    let mut deleted_any = false;
    if let Some(cache_dir) = canonical.parent() {
        if cache_dir.exists() {
            let count = walk_file_count(cache_dir);
            std::fs::remove_dir_all(cache_dir)?;
            println!("Deleted cache directory {} ({} files)", cache_dir.display(), count);
            deleted_any = true;
        } else if canonical.exists() {
            std::fs::remove_file(canonical)?;
            println!("Deleted {}", canonical.display());
            deleted_any = true;
        } else {
            println!("No cache found at {} (already deleted?)", cache_dir.display());
        }
    } else if canonical.exists() {
        std::fs::remove_file(canonical)?;
        println!("Deleted {}", canonical.display());
        deleted_any = true;
    }
    if let Some(out) = output {
        if out.exists() {
            std::fs::remove_file(out)?;
            println!("Deleted output {}", out.display());
            deleted_any = true;
        } else {
            println!("Output file not found: {} (already deleted?)", out.display());
        }
    }
    if !deleted_any {
        println!("Nothing to delete.");
    }
    Ok(())
}

fn walk_file_count(dir: &std::path::Path) -> usize {
    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += walk_file_count(&path);
            } else {
                count += 1;
            }
        }
    }
    count
}


// Generic convert/delete that dispatch by model name — mirrors `zsfm <model> convert/delete`.
#[pyfunction]
#[pyo3(signature = (model, output=None, dtype="f16", model_dir="models", token=None, redownload=false, task=None, filename=None))]
fn convert(
    model: &str,
    output: Option<String>,
    dtype: &str,
    model_dir: &str,
    token: Option<String>,
    redownload: bool,
    task: Option<String>,
    filename: Option<String>,
) -> PyResult<()> {
    let dtype_ty = dtype_from_str(dtype)?;
    let model_dir = PathBuf::from(model_dir);
    let rt = tokio::runtime::Runtime::new().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    rt.block_on(async {
        match model {
            "toto" => {
                let repo = "Datadog/Toto-2.0-2.5B";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/toto-2.5b-f16.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_toto::config::TotoConfig::from_json(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                zsfm_toto::convert::convert(repo, &files, &cfg, &zsfm_toto::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok::<(), anyhow::Error>(())
            }
            "chronos" => {
                let repo = "amazon/chronos-2";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/chronos-f16.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_chronos::config::Chronos2Config::from_json(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                zsfm_chronos::convert::convert(repo, &files, &cfg, &zsfm_chronos::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "timesfm" => {
                let repo = "google/timesfm-2.5-200m-pytorch";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/timesfm.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_timesfm::config::TimesFMConfig::new();
                zsfm_timesfm::convert::convert(repo, &files, &cfg, &zsfm_timesfm::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "sundial" => {
                let repo = "thuml/sundial-base-128m";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/sundial-f16.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg: zsfm_sundial::config::SundialConfig = serde_json::from_str(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                zsfm_sundial::convert::convert(repo, &files, &cfg, &zsfm_sundial::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "ttm" => {
                let repo = "ibm-granite/granite-timeseries-ttm-r2";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/ttm-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_ttm::config::TtmConfig::from_json(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                zsfm_ttm::convert::convert(repo, &files, &cfg, &zsfm_ttm::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "lag_llama" | "lag-llama" => {
                let repo = "time-series-foundation-models/Lag-Llama";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/lag_llama-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                if let Some(p) = canonical.parent() { std::fs::create_dir_all(p).map_err(|e| anyhow::anyhow!(e.to_string()))?; }
                let ckpt = zsfm_hub::download_file(repo, "lag-llama.ckpt", token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_lag_llama::config::LagLlamaConfig::default_from_ckpt();
                zsfm_lag_llama::convert::convert(&ckpt, &cfg, &zsfm_lag_llama::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let _ = std::fs::remove_file(&ckpt);
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "moment" => {
                let repo = "AutonLab/MOMENT-1-large";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/moment-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_moment::config::MomentConfig::default();
                zsfm_moment::convert::convert(&files.safetensors_shards, &cfg, &zsfm_moment::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "moirai" => {
                let repo = "Salesforce/moirai-1.0-R-large";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/moirai-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_moirai::config::MoiraiConfig::default();
                zsfm_moirai::convert::convert(&files.safetensors_shards, &cfg, &zsfm_moirai::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "moirai2" | "moirai-2" => {
                let repo = "Salesforce/moirai-2.0-R-small";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/moirai2-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_moirai2::config::Moirai2Config::default();
                zsfm_moirai2::convert::convert(&files.safetensors_shards, &cfg, &zsfm_moirai2::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "flowstate" | "flowstate-r1" => {
                let repo = "ibm-granite/granite-timeseries-flowstate-r1";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/flowstate-r1-f16.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_flowstate::config::FlowStateConfig::from_json(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                zsfm_flowstate::convert::convert(repo, &files, &cfg, &zsfm_flowstate::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "tirex" => {
                let repo = "NX-AI/TiRex";
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("gguf/tirex-f32.gguf"));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                if let Some(p) = canonical.parent() { std::fs::create_dir_all(p).map_err(|e| anyhow::anyhow!(e.to_string()))?; }
                let ckpt = zsfm_hub::download_file(repo, "model.ckpt", token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_tirex::config::TiRexConfig::default_from_ckpt();
                zsfm_tirex::convert::convert(&ckpt, &cfg, &zsfm_tirex::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let _ = std::fs::remove_file(&ckpt);
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "mitra" => {
                let task_str = task.as_deref().unwrap_or("classification");
                let repo = if task_str == "regression" { "autogluon/mitra-regressor" } else { "autogluon/mitra-classifier" };
                let variant_dir = model_dir.join(format!("mitra-{task_str}"));
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("gguf/mitra-{task_str}-{dtype}.gguf")));
                let canonical = zsfm_hub::canonical_gguf_path(&variant_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model(repo, token.as_deref(), &variant_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = if task_str == "regression" { zsfm_mitra::config::MitraConfig::regressor() } else { zsfm_mitra::config::MitraConfig::classifier() };
                let p = files.safetensors_shards.first().ok_or_else(|| anyhow::anyhow!("no shard"))?;
                zsfm_mitra::convert::convert(std::slice::from_ref(p), &cfg, &zsfm_mitra::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "tabdpt" => {
                let repo = "Layer6/TabDPT";
                let fname = filename.as_deref().unwrap_or("tabdpt1_2.safetensors");
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("gguf/tabdpt-{dtype}.gguf")));
                let canonical = zsfm_hub::canonical_gguf_path(&model_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                if let Some(p) = canonical.parent() { std::fs::create_dir_all(p).map_err(|e| anyhow::anyhow!(e.to_string()))?; }
                let p = zsfm_hub::download_file(repo, fname, token.as_deref(), &model_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_tabdpt::config::TabDptConfig::default_v1_2();
                zsfm_tabdpt::convert::convert(std::slice::from_ref(&p), &cfg, &zsfm_tabdpt::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let _ = std::fs::remove_file(&p);
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            "tabfm" => {
                let task_str = task.as_deref().unwrap_or("classification");
                let repo = "google/tabfm-1.0.0-pytorch";
                let variant_dir = model_dir.join(format!("tabfm-{task_str}"));
                let out = output.map(PathBuf::from).unwrap_or_else(|| PathBuf::from(format!("gguf/tabfm-{task_str}-{dtype}.gguf")));
                let canonical = zsfm_hub::canonical_gguf_path(&variant_dir, repo);
                if canonical.exists() && !redownload {
                    zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                    return Ok(());
                }
                let files = zsfm_hub::download_model_prefixed(repo, task_str, token.as_deref(), &variant_dir).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let s = std::fs::read_to_string(&files.config_json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let cfg = zsfm_tabfm::config::TabFMConfig::from_json(&s).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let p = files.safetensors_shards.first().ok_or_else(|| anyhow::anyhow!("no shard"))?;
                zsfm_tabfm::convert::convert(repo, p, &cfg, &zsfm_tabfm::convert::ConvertOptions { output_dtype: zsfm_gguf::GGMLType::F32 }, &canonical).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                files.cleanup_weights();
                zsfm_checkpoint::recast(&canonical, &out, dtype_ty).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                Ok(())
            }
            other => Err(anyhow::anyhow!("unknown model {other:?}; expected one of {}", list_models().join(", ")))
        }
    }).map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (model, model_dir="models", output=None))]
fn delete(model: &str, model_dir: &str, output: Option<String>) -> PyResult<()> {
    // Map friendly names to canonical repo ids for per-model delete
    let (repo, variant_dir): (String, Option<String>) = match model {
        "toto" => ("Datadog/Toto-2.0-2.5B".into(), None),
        "chronos" => ("amazon/chronos-2".into(), None),
        "timesfm" => ("google/timesfm-2.5-200m-pytorch".into(), None),
        "sundial" => ("thuml/sundial-base-128m".into(), None),
        "ttm" => ("ibm-granite/granite-timeseries-ttm-r2".into(), None),
        "lag_llama" | "lag-llama" => ("time-series-foundation-models/Lag-Llama".into(), None),
        "moment" => ("AutonLab/MOMENT-1-large".into(), None),
        "moirai" => ("Salesforce/moirai-1.0-R-large".into(), None),
        "moirai2" | "moirai-2" => ("Salesforce/moirai-2.0-R-small".into(), None),
        "flowstate" | "flowstate-r1" => ("ibm-granite/granite-timeseries-flowstate-r1".into(), None),
        "tirex" => ("NX-AI/TiRex".into(), None),
        "mitra" | "mitra-classification" => ("autogluon/mitra-classifier".into(), Some("mitra-classification".into())),
        "mitra-regression" => ("autogluon/mitra-regressor".into(), Some("mitra-regression".into())),
        "tabdpt" => ("Layer6/TabDPT".into(), None),
        "tabicl" => ("jingang/TabICL".into(), None),
        "tabpfn" => ("Prior-Labs/tabpfn_3".into(), None),
        "tabfm" | "tabfm-classification" => ("google/tabfm-1.0.0-pytorch".into(), Some("tabfm-classification".into())),
        "tabfm-regression" => ("google/tabfm-1.0.0-pytorch".into(), Some("tabfm-regression".into())),
        // also accept raw repo ids
        _ if model.contains('/') => (model.to_string(), None),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown model {model:?}"))),
    };
    let model_dir = PathBuf::from(model_dir);
    let canonical = if let Some(v) = variant_dir {
        let vd = model_dir.join(v);
        zsfm_hub::canonical_gguf_path(&vd, &repo)
    } else {
        zsfm_hub::canonical_gguf_path(&model_dir, &repo)
    };
    let out = output.map(PathBuf::from);
    delete_cached_model(&canonical, out.as_deref()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Forecasters — one #[pyclass] per model (point or quantile)
// ---------------------------------------------------------------------------

use zsfm_toto::config::TotoConfig;
use zsfm_toto::infer::TotoModel as RustTotoModel;

#[pyclass]
struct TotoModel {
    inner: RustTotoModel,
    _gguf: String,
}

#[pymethods]
impl TotoModel {
    #[new]
    #[pyo3(signature = (gguf, config=None, context_length=None, use_f64=false))]
    fn new(gguf: String, config: Option<String>, context_length: Option<usize>, use_f64: bool) -> PyResult<Self> {
        let gguf_path = PathBuf::from(&gguf);
        let cfg_path = config.unwrap_or_else(|| "models/Datadog__Toto-2.0-2.5B/config.json".into());
        let s = std::fs::read_to_string(&cfg_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{cfg_path}: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let max_ctx = context_length.unwrap_or(4096);
        let inner = RustTotoModel::builder(&gguf_path)
            .config_json(&v)
            .with_compute_f64(use_f64)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Toto {gguf}: {e}")))?;
        // Store max_ctx via a wrapper? For now keep simple: the Rust model already encodes patch_size; we trim in forecast.
        let _ = max_ctx;
        Ok(Self { inner, _gguf: gguf })
    }
    /// Forecast from a univariate context. For batch/multivariate, call
    /// `forecast_batch` (or loop). Returns the point forecast (median).
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let data = vec![context.clone()];
        let mask = vec![vec![true; context.len()]];
        let qmat = self.inner.forecast(&data, &mask, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let median_idx = 4; // q0.5
        Ok(qmat[median_idx][0].clone())
    }

    /// Batch forecast: context is List[List[float]] with shape [batch][time].
    fn forecast_batch(&self, contexts: Vec<Vec<f32>>, horizon: usize) -> PyResult<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        for ctx in contexts {
            let data = vec![ctx.clone()];
            let mask = vec![vec![true; ctx.len()]];
            let qmat = self.inner.forecast(&data, &mask, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            out.push(qmat[4][0].clone());
        }
        Ok(out)
    }
}

use zsfm_timesfm::infer::TimesFMModel as RustTimesFmModel;
#[pyclass]
struct TimesFmModel { inner: RustTimesFmModel }
#[pymethods]
impl TimesFmModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let inner = RustTimesFmModel::load(&p).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TimesFM {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out.into_iter().next().unwrap_or_default())
    }
}

use zsfm_sundial::infer::SundialModel as RustSundialModel;
#[pyclass]
struct SundialModel { inner: RustSundialModel }
#[pymethods]
impl SundialModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let inner = RustSundialModel::builder(&p).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Sundial {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let raw = self.inner.forecast(&context, &candle_core::Device::Cpu).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(raw.into_iter().take(horizon).collect())
    }
}

use zsfm_ttm::config::TtmConfig;
use zsfm_ttm::infer::TtmModel as RustTtmModel;

#[pyclass]
struct TtmModel {
    inner: RustTtmModel,
    _config_path: Option<String>,
}

#[pymethods]
impl TtmModel {
    #[new]
    #[pyo3(signature = (gguf, config=None))]
    fn new(gguf: String, config: Option<String>) -> PyResult<Self> {
        let gguf_path = PathBuf::from(&gguf);
        let ttm_config = if let Some(cfg_path) = &config {
            let s = std::fs::read_to_string(cfg_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read config {cfg_path}: {e}")))?;
            TtmConfig::from_json(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            let canonical = PathBuf::from("models/ibm-granite__granite-timeseries-ttm-r2/config.json");
            if canonical.exists() {
                let s = std::fs::read_to_string(&canonical).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                TtmConfig::from_json(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            } else {
                return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!("config not found at {canonical:?} and no `config` arg given")));
            }
        };
        let inner = RustTtmModel::builder(&gguf_path).config(ttm_config).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TTM {gguf}: {e}")))?;
        Ok(Self { inner, _config_path: config })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out.into_iter().take(horizon).collect())
    }
}

use zsfm_lag_llama::config::LagLlamaConfig;
use zsfm_lag_llama::infer::LagLlamaModel as RustLagLlamaModel;
#[pyclass]
struct LagLlamaModel { inner: RustLagLlamaModel }
#[pymethods]
impl LagLlamaModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = LagLlamaConfig::default_from_ckpt();
        let inner = RustLagLlamaModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load LagLlama {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_moment::config::MomentConfig;
use zsfm_moment::infer::MomentModel as RustMomentModel;
#[pyclass]
struct MomentModel { inner: RustMomentModel }
#[pymethods]
impl MomentModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = MomentConfig::default();
        let inner = RustMomentModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Moment {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_moirai::config::MoiraiConfig;
use zsfm_moirai::infer::MoiraiModel as RustMoiraiModel;
#[pyclass]
struct MoiraiModel { inner: RustMoiraiModel }
#[pymethods]
impl MoiraiModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = MoiraiConfig::default();
        let inner = RustMoiraiModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Moirai {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_moirai2::config::Moirai2Config;
use zsfm_moirai2::infer::Moirai2Model as RustMoirai2Model;
#[pyclass]
struct Moirai2Model { inner: RustMoirai2Model }
#[pymethods]
impl Moirai2Model {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = Moirai2Config::default();
        let inner = RustMoirai2Model::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Moirai2 {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let out = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_flowstate::config::FlowStateConfig;
use zsfm_flowstate::infer::FlowStateModel as RustFlowStateModel;
#[pyclass]
struct FlowStateModel { inner: RustFlowStateModel }
#[pymethods]
impl FlowStateModel {
    #[new]
    #[pyo3(signature = (gguf, config=None))]
    fn new(gguf: String, config: Option<String>) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg_path = config.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("models/ibm-granite__granite-timeseries-flowstate-r1/config.json"));
        let s = std::fs::read_to_string(&cfg_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {}: {e}", cfg_path.display())))?;
        let cfg = FlowStateConfig::from_json(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let inner = RustFlowStateModel::builder(&p).config_from(&cfg).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load FlowState {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let qmat = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let median_idx = self.inner.config.median_index();
        Ok(qmat[median_idx].clone())
    }
}

use zsfm_tirex::config::TiRexConfig;
use zsfm_tirex::infer::TiRexModel as RustTirexModel;
#[pyclass]
struct TirexModel { inner: RustTirexModel }
#[pymethods]
impl TirexModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = TiRexConfig::default_from_ckpt();
        let inner = RustTirexModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TiRex {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let ( _q, mean) = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(mean)
    }
}

use zsfm_chronos::config::Chronos2Config;
use zsfm_chronos::infer::ChronosModel as RustChronosModel;

#[pyclass]
struct ChronosModel { inner: RustChronosModel }
#[pymethods]
impl ChronosModel {
    #[new]
    #[pyo3(signature = (gguf, config=None))]
    fn new(gguf: String, config: Option<String>) -> PyResult<Self> {
        let gguf_path = PathBuf::from(&gguf);
        let cfg_path = config.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("models/amazon__chronos-2/config.json"));
        let s = std::fs::read_to_string(&cfg_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {}: {e}", cfg_path.display())))?;
        let cfg = Chronos2Config::from_json(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let inner = RustChronosModel::builder(&gguf_path).config_from(&cfg).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Chronos {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn forecast(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<f32>> {
        let qmat = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let levels = self.inner.config.quantiles();
        let median_idx = levels.iter().position(|&q| (q - 0.5).abs() < 1e-6).unwrap_or(levels.len()/2);
        Ok(qmat[median_idx].clone())
    }
    fn forecast_quantiles(&self, context: Vec<f32>, horizon: usize) -> PyResult<Vec<Vec<f32>>> {
        let qmat = self.inner.forecast(&context, horizon).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(qmat)
    }
    fn quantiles(&self) -> Vec<f32> { self.inner.config.quantiles().to_vec() }
}

// ---------------------------------------------------------------------------
// Tabular models
// ---------------------------------------------------------------------------

use zsfm_mitra::config::MitraConfig;
use zsfm_mitra::MitraModel as RustMitraModel;
#[pyclass]
struct MitraModel { inner: RustMitraModel, is_classifier: bool }
#[pymethods]
impl MitraModel {
    #[new]
    #[pyo3(signature = (gguf, task="classification"))]
    fn new(gguf: String, task: &str) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = match task {
            "classification" => MitraConfig::classifier(),
            "regression" => MitraConfig::regressor(),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("task must be 'classification' or 'regression'"))
        };
        let is_classifier = task == "classification";
        let inner = RustMitraModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load Mitra {gguf}: {e}")))?;
        Ok(Self { inner, is_classifier })
    }
    fn predict_classification(&self, x_support: Vec<Vec<f32>>, y_support: Vec<usize>, x_query: Vec<Vec<f32>>, n_classes: usize) -> PyResult<Vec<Vec<f32>>> {
        if !self.is_classifier { return Err(pyo3::exceptions::PyValueError::new_err("model was loaded as regressor, not classifier")) }
        let logits = self.inner.predict_classification(&x_support, &y_support, &x_query, n_classes).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(logits)
    }
    fn predict_regression(&self, x_support: Vec<Vec<f32>>, y_support: Vec<f32>, x_query: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        if self.is_classifier { return Err(pyo3::exceptions::PyValueError::new_err("model was loaded as classifier, not regressor")) }
        let out = self.inner.predict_regression(&x_support, &y_support, &x_query).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_tabdpt::config::TabDptConfig;
use zsfm_tabdpt::TabDptModel as RustTabDptModel;
#[pyclass]
struct TabDptModel { inner: RustTabDptModel }
#[pymethods]
impl TabDptModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = TabDptConfig::default_v1_2();
        let inner = RustTabDptModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TabDPT {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn predict_classification(&self, x_support: Vec<Vec<f32>>, y_support: Vec<usize>, x_query: Vec<Vec<f32>>, n_classes: usize) -> PyResult<Vec<Vec<f32>>> {
        let out = self.inner.predict_classification(&x_support, &y_support, &x_query, n_classes).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
    fn predict_regression(&self, x_support: Vec<Vec<f32>>, y_support: Vec<f32>, x_query: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
        let out = self.inner.predict_regression(&x_support, &y_support, &x_query).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_tabicl::config::TabIclConfig;
use zsfm_tabicl::TabIclModel as RustTabIclModel;
#[pyclass]
struct TabIclModel { inner: RustTabIclModel }
#[pymethods]
impl TabIclModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = TabIclConfig::v2();
        let inner = RustTabIclModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TabICL {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn predict_classification(&self, x_support: Vec<Vec<f32>>, y_support: Vec<usize>, x_query: Vec<Vec<f32>>, n_classes: usize) -> PyResult<Vec<Vec<f32>>> {
        let out = self.inner.predict_classification(&x_support, &y_support, &x_query, n_classes).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_tabpfn::config::TabPfnConfig;
use zsfm_tabpfn::TabPfnModel as RustTabPfnModel;
#[pyclass]
struct TabPfnModel { inner: RustTabPfnModel }
#[pymethods]
impl TabPfnModel {
    #[new]
    fn new(gguf: String) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg = TabPfnConfig::v3_default();
        let inner = RustTabPfnModel::load(&p, cfg).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TabPFN {gguf}: {e}")))?;
        Ok(Self { inner })
    }
    fn predict_classification(&self, x_support: Vec<Vec<f32>>, y_support: Vec<usize>, x_query: Vec<Vec<f32>>, n_classes: usize) -> PyResult<Vec<Vec<f32>>> {
        let out = self.inner.predict_classification(&x_support, &y_support, &x_query, n_classes).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}

use zsfm_tabfm::config::TabFMConfig;
use zsfm_tabfm::TabFMModel as RustTabFMModel;
#[pyclass]
struct TabFmModel { inner: RustTabFMModel, is_classifier: bool }
#[pymethods]
impl TabFmModel {
    #[new]
    #[pyo3(signature = (gguf, config=None))]
    fn new(gguf: String, config: Option<String>) -> PyResult<Self> {
        let p = PathBuf::from(&gguf);
        let cfg_path = if let Some(c) = config { PathBuf::from(c) } else {
            // Auto-detect: try classification then regression config
            let cand = PathBuf::from("models/tabfm-classification/google__tabfm-1.0.0-pytorch/classification_config.json");
            if cand.exists() { cand } else { PathBuf::from("models/tabfm-regression/google__tabfm-1.0.0-pytorch/regression_config.json") }
        };
        let s = std::fs::read_to_string(&cfg_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {}: {e}", cfg_path.display())))?;
        let cfg = TabFMConfig::from_json(&s).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let is_classifier = cfg.is_classifier;
        let inner = RustTabFMModel::builder(&p).config_from(&cfg).build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("load TabFM {gguf}: {e}")))?;
        Ok(Self { inner, is_classifier })
    }
    fn predict(&self, x: Vec<Vec<f32>>, y: Vec<f32>, train_size: usize) -> PyResult<Vec<Vec<f32>>> {
        let out = self.inner.predict(&x, &y, train_size, None, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(out)
    }
}
