# Development

## Workspace layout

```
.
├── Cargo.toml         # root workspace (so `cargo install --git https://github.com/amaye15/zsfm-rs zsfm` works)
├── pyproject.toml     # Python project (uv + maturin + pyo3, module `zsfm`)
└── zsfm-rs/
    crates/
      zsfm/            # the `zsfm` binary — one subcommand module per model (`cargo install zsfm`)
      zsfm-python/     # `zsfm` Python extension (pyo3, `import zsfm`; see ./python.md)
      zsfm-gguf/        # GGUF reader/writer
      zsfm-checkpoint/  # loads safetensors/pickle/onnx/hdf5/npz/ckpt/gguf, dtype casting, recast()
      zsfm-hub/         # HuggingFace download + the canonical-F32-cache helpers
      zsfm-nn/          # shared candle tensor primitives
      zsfm-bench/       # in-process rolling-window accuracy/latency benchmark + ensembling
      models/
        chronos/ flowstate/ moirai/ moirai2/ moment/ sundial/ timesfm/ toto/ ttm/
        lag_llama/ tirex/                          # time-series forecasters
        mitra/ tabdpt/ tabicl/ tabpfn/ tabfm/       # tabular foundation models
```

Each model crate under `models/` owns its architecture, weight-name mapping, and inference kernel; `zsfm` (`crates/zsfm/`) just wires them up to `convert`/`infer`/`upload`/`inspect-tensors`/`delete` subcommands.

## Building and testing

```bash
# Rust — from the repo root (uses the root workspace, which re-exports zsfm-rs):
cargo build --release --workspace
cargo test --release --workspace
# or, from inside zsfm-rs (same result, uses zsfm-rs/Cargo.toml):
cd zsfm-rs
cargo build --release --workspace
cargo test --release --workspace
# also: cargo install from crates.io or git, like ripgrep:
cargo install zsfm --locked
cargo install --git https://github.com/amaye15/zsfm-rs zsfm --locked

# Python — uv + pyo3 + maturin (see ./python.md)
uv sync
cargo build --release -p zsfm-python   # check Rust alone
uv run maturin develop                 # build + install as editable (fastest)
uv run pytest tests/python -v          # or: .venv/bin/python -m pytest
uv run python -c "import zsfm; print(zsfm.list_models())"
```

The baseline is 130 passing tests across the workspace (unit tests for tensor casting, GGUF round-tripping, per-model architecture/shape checks, and `zsfm-bench`'s window-generation/metrics/ensembling logic) plus 6 Python tests (`tests/python/test_zsfm.py`). CI (`.github/workflows/ci.yml`) runs both on every push to `main` and every PR, on Linux and macOS — Rust (`cargo build`/`cargo test`) and Python (`uv run maturin develop` + `pytest`).

## Verifying a model port is correct

Every model in this workspace was verified **bit-exact** (or numerically equivalent within float tolerance) against its original PyTorch implementation before being considered done — same input, same output, checkpoint tensor-for-tensor. If you're modifying a model crate, re-run that model's `convert` + `infer` against a known input/output pair before assuming a change is safe; there isn't a single workspace-wide golden-output test harness, so this is a manual step per model.

## Benchmarking

`zsfm-bench` (`crates/zsfm-bench`) runs the rolling-window accuracy/latency benchmark across the 11 time-series forecasters and writes `benchmark.md`. It links the model crates in-process through the shared `zsfm_core::Forecaster` interface — each model's GGUF is loaded exactly once and reused across every dataset/window/context/horizon combination, and independent models run concurrently — rather than the old Python driver's one-subprocess-and-reload-the-model-every-time approach.

```bash
# convert whichever models you want to benchmark first, e.g.:
zsfm ttm convert && zsfm moirai2 convert

# one dataset, quick look:
zsfm-bench run --dataset ETTh1 --models ttm,moirai2 --windows 30

# full 21-dataset × horizon/context sweep, writes ../benchmark.md:
zsfm-bench report
```

`report` caches each (dataset, context, horizon, windows) config's results in `benchmark/bench_cache.json` (gitignored) — re-running after an interruption, or after adding a model, only computes what's missing.

## Adding a new model

Roughly the shape to follow, based on the existing crates under `models/`:

1. New crate under `crates/models/<name>/` with a weight-name mapping from the original checkpoint to whatever internal names you want, an inference module built on candle, and a request/response type.
2. A `zsfm/src/<name>.rs` (`zsfm-rs/crates/zsfm/src/<name>.rs`) with `convert`/`infer`/`delete` subcommands, following the caching pattern described in [The `zsfm` CLI](./cli-overview.md#model-caching) — compute `zsfm_hub::canonical_gguf_path`, check it before downloading, call `zsfm_checkpoint::recast` for cache hits and for the final requested dtype after a fresh download, and implement `delete` via `common::delete_cached_model`.
3. Wire the subcommand into `zsfm`'s top-level `Commands` enum (`zsfm-rs/crates/zsfm/src/main.rs`).
4. A page in this guide (`docs-guide/src/models/`) and a row in the relevant summary table.

## Releasing

Pushing a `v*` tag triggers `.github/workflows/release.yml`: cross-platform binary builds (`cargo build --release -p zsfm`), a GitHub Release with `--generate-notes`, and (gated behind the `PUBLISH_CRATES_IO` repository variable) publishing all 22 crates to crates.io in dependency order via `zsfm-rs/scripts/publish-crates.sh` (`cargo publish -p zsfm` is last).
