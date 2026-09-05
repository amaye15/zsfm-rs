# Python bindings (uv + pyo3 + maturin)

The same 16 models are available from Python via `zsfm` — installed with [`uv`](https://docs.astral.sh/uv/) and built with [`maturin`](https://github.com/PyO3/maturin) + [`pyo3`](https://pyo3.rs). The API mirrors the CLI (`convert / infer / delete`) but as Python classes/functions, with `numpy` arrays for inputs/outputs where natural.

## Installation

Requires Python ≥3.8 and a recent Rust toolchain (`rustup.rs`).

```bash
# from crates.io (once published)
uv pip install zsfm
# or: pip install zsfm

# from git (latest main)
uv pip install "zsfm @ git+https://github.com/amaye15/zsfm-rs"

# from a local checkout (editable, fastest for development)
git clone https://github.com/amaye15/zsfm-rs.git
cd zsfm-rs   # repo root is zero-shot-forecasters-gguf
uv sync                    # creates .venv, installs deps + zsfm as editable
uv run maturin develop     # rebuild after Rust changes (or: maturin develop)
# alternative without uv: pip install -e .
```

> The Rust code lives under `zsfm-rs/` but the Python project root is the repo
> root (where `pyproject.toml` lives). `uv sync` + `uv run maturin develop`
> is the `uv` analogue of `cargo install zsfm` for Python.

Verify:

```bash
uv run python -c "import zsfm; print(zsfm.__version__); print(zsfm.list_models())"
# ['toto', 'chronos', 'timesfm', 'sundial', 'ttm', 'lag_llama', 'moment', 'moirai', 'moirai2', 'flowstate', 'tirex', 'mitra', 'tabdpt', 'tabicl', 'tabpfn', 'tabfm']
```

## Quick start

### Forecasting (time series)

```python
import zsfm

# list all models
print(zsfm.list_forecasters())
# ['toto', 'chronos', 'timesfm', 'sundial', 'ttm', 'lag_llama', 'moment', 'moirai', 'moirai2', 'flowstate', 'tirex']

# download + convert (like `zsfm ttm convert`)
zsfm.convert("ttm", output="gguf/ttm-f32.gguf", dtype="f32")  # also: model_dir, token, redownload, task/filename
# or per-model: already handled by the generic `convert` dispatcher

# load and forecast (like `zsfm ttm infer --gguf gguf/ttm-f32.gguf`)
model = zsfm.TtmModel("gguf/ttm-f32.gguf", config="models/ibm-granite__granite-timeseries-ttm-r2/config.json")
context = [10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 12.2]
point = model.forecast(context, horizon=4)
print(point)  # [12.3, 12.5, 12.6, 12.7]

# Chronos-2 exposes quantiles as well
chronos = zsfm.ChronosModel("gguf/chronos-f16.gguf")
qmat = chronos.forecast_quantiles([1,2,3,4,5,6,7,8], horizon=4)  # [n_quantiles][horizon]
print(chronos.quantiles())  # [0.1, 0.2, ..., 0.9]

# Toto / Moirai support batch + multivariate (mirrors CLI JSON shapes)
# For now, Python's TotoModel exposes `forecast` (single series) and `forecast_batch`
# (List[List[float]] -> List[List[float]]).

# delete cache (like `zsfm ttm delete`)
zsfm.delete("ttm")  # removes models/ibm-granite__granite-timeseries-ttm-r2/
zsfm.delete("ttm", output="gguf/ttm-f32.gguf")  # also remove the converted file
```

### Tabular (zero-shot classification / regression)

```python
import zsfm

# Mitra (needs task)
mitra_clf = zsfm.MitraModel("gguf/mitra-classification-f32.gguf", task="classification")
logits = mitra_clf.predict_classification(
    x_support=[[0.1, 1.2], [0.9, -0.3]],
    y_support=[0, 1],
    x_query=[[0.2, 0.9]],
    n_classes=2,
)
print(logits)  # [[...], [...]]

mitra_reg = zsfm.MitraModel("gguf/mitra-regression-f32.gguf", task="regression")
preds = mitra_reg.predict_regression(
    x_support=[[0.1], [0.9]],
    y_support=[0.5, 1.5],
    x_query=[[0.2]],
)
print(preds)

# TabDPT (single checkpoint, task at predict time)
tabdpt = zsfm.TabDptModel("gguf/tabdpt-f32.gguf")
probs = tabdpt.predict_classification(x_support, y_support, x_query, n_classes=2)

# TabICL / TabPFN-3 (classification only)
tabicl = zsfm.TabIclModel("gguf/tabicl-v2-f32.gguf")
tabpfn = zsfm.TabPfnModel("gguf/tabpfn-v3-f32.gguf")

# TabFM (needs config, like the CLI)
tabfm = zsfm.TabFmModel("gguf/tabfm-classification-f16.gguf")
# single predict (like `zsfm tabfm infer`)
out = tabfm.predict(x=[[0.1, 1.2], [0.9, -0.3]], y=[0, 1], train_size=1)
# ensemble-predict is not yet exposed in Python; use the CLI for now:
# zsfm tabfm ensemble-predict --gguf ... < request.json
```

For exact per-model `convert` defaults and GGUF paths, see `zsfm <model> convert --help` — the Python `convert(model, ...)` dispatcher accepts the same `model`, `model_dir`, `output`, `dtype`, `token`, `redownload`, `task`, `filename` kwargs.

## API reference

Top-level:

| Symbol | Kind | Description |
|---|---|---|
| `zsfm.__version__` | `str` | Crate version (matches `Cargo.toml` workspace version) |
| `zsfm.list_models()` | `fn -> List[str]` | All 16 model ids |
| `zsfm.list_forecasters()` | `fn -> List[str]` | 11 forecasters |
| `zsfm.list_tabular()` | `fn -> List[str]` | 5 tabular |
| `zsfm.convert(model, output?, dtype?, model_dir?, token?, redownload?, task?, filename?)` | `fn` | Download + convert (dispatches by model id, like `zsfm <model> convert`) |
| `zsfm.delete(model, model_dir?, output?)` | `fn` | Remove cache (like `zsfm <model> delete`) |

Forecasters (each `Model(gguf, config?)` with `forecast(context, horizon)`):

`TotoModel`, `ChronosModel`, `TimesFmModel`, `SundialModel`, `TtmModel`, `LagLlamaModel`, `MomentModel`, `MoiraiModel`, `Moirai2Model`, `FlowStateModel`, `TirexModel`

- `TotoModel(gguf, config?, context_length?, use_f64?)` — also `forecast_batch`
- `ChronosModel(gguf, config?)` — also `forecast_quantiles`, `quantiles()`
- `FlowStateModel(gguf, config?)` / `TtmModel(gguf, config?)` / `ChronosModel` — `config` is `models/<owner>__<name>/config.json` from `convert`
- Others with fixed configs: `TimesFmModel(gguf)`, `SundialModel(gguf)`, `LagLlamaModel(gguf)`, `MomentModel(gguf)`, `MoiraiModel(gguf)`, `Moirai2Model(gguf)`, `TirexModel(gguf)`

Tabular:

`MitraModel(gguf, task?)` — `predict_classification` / `predict_regression`  
`TabDptModel(gguf)` — `predict_classification` / `predict_regression`  
`TabIclModel(gguf)` — `predict_classification`  
`TabPfnModel(gguf)` — `predict_classification`  
`TabFmModel(gguf, config?)` — `predict(x, y, train_size)` (single forward pass)

All `forecast`/`predict` methods accept Python `list` or `numpy.ndarray` and return `list` (convert to `numpy` via `np.array(...)` if you prefer).

## Development

```bash
# Rust + Python together
uv sync                          # (re)create .venv with deps
cargo build --release -p zsfm-python  # check Rust alone
uv run maturin develop           # build + install as editable (fastest)
# or: maturin develop --manifest-path zsfm-rs/crates/zsfm-python/Cargo.toml

# run Python tests
uv run pytest tests/python -v
# or: .venv/bin/python -m pytest

# from crates.io (once published)
cargo publish -p zsfm-python   # last, after the other 22 crates
```

`pyproject.toml` at the repo root is the `uv`/`maturin` project ( `tool.maturin.manifest-path = "zsfm-rs/crates/zsfm-python/Cargo.toml"`, `module-name = "zsfm"` ). The Rust workspace at `zsfm-rs/Cargo.toml` and the root `Cargo.toml` both include `zsfm-python` as a member so `cargo install --git` and `cargo build --workspace` keep working.

See also the `zsfm` API docs at `https://amaye15.github.io/zsfm-rs/api/zsfm_python/` (once `cargo doc` includes the pyo3 crate).
