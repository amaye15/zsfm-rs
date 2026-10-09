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

The baseline is 182 passing tests across the workspace (unit tests for tensor casting, GGUF round-tripping, per-model architecture/shape checks, and `zsfm-bench`'s window-generation/metrics/ensembling logic) plus 6 Python tests (`tests/python/test_zsfm.py`). CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `cargo clippy -D warnings`, Rust (`cargo build`/`cargo test`) and Python (`uv run maturin develop` + `pytest`) on every push to `main` and every PR, on Linux and macOS, plus a Windows `cargo check` and `cargo audit`.

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

1. New crate under `crates/models/<name>/` with a weight-name mapping from the original checkpoint to whatever internal names you want, an inference module built on candle, and a request/response type. Put exact-match name pairs in a `TABLE` and look them up with `zsfm_gguf::map_with_table`; keep only prefix/layer rules as code (see `models/timesfm/src/tensor_map.rs`). Share norm/softmax/attention/RoPE via `zsfm_nn` instead of private copies.
2. A `zsfm/src/<name>.rs` (`zsfm-rs/crates/zsfm/src/<name>.rs`) with `convert`/`infer`/`delete` subcommands, following the caching pattern described in [The `zsfm` CLI](./cli-overview.md#model-caching) — compute `zsfm_hub::canonical_gguf_path` (or `variant_gguf_path` for multi-task repos), check it via `zsfm_hub::try_recast_from_cache` before downloading, call `zsfm_checkpoint::recast` for the final dtype, use `crate::common::DtypeArg` (not a local copy), read stdin via `zsfm_core::read_stdin_limited`, parse horizon via `zsfm_core::parse_horizon`, and implement `delete` via `common::delete_cached_model`.
3. Wire the subcommand into `zsfm`'s top-level `Commands` enum (`zsfm-rs/crates/zsfm/src/main.rs`). Look up the default repo in `zsfm_hub::MODEL_REPOS` so CLI and Python cannot drift.
4. A page in this guide (`docs-guide/src/models/`) and a row in the relevant summary table.

## Platform notes

- macOS links Apple's Accelerate via `candle` `accelerate` (enabled only in `crates/zsfm` + `zsfm-bench` for `cfg(target_os = "macos")`; other crates inherit it through feature unification when built as part of the binary). Linux/Windows use candle's portable backend — correct but 2-5x slower on GEMM. For native-CPU Linux builds, uncomment `target-cpu=native` in `.cargo/config.toml`.
- Windows builds: `cargo check` runs in CI (`windows-check` job). Full workspace needs `hdf5-metno` C build which often fails on MSVC; at minimum `zsfm-core`, `zsfm-gguf`, `zsfm-nn`, `zsfm-hub` check cleanly. No release artifact is published for Windows yet.
- TiRex ships a `.ckpt` (not safetensors), so `zsfm-tirex` intentionally omits the `safetensors` dependency and goes through the generic checkpoint path.
- Downloads stream with resume + `Content-Range` and byte-count checks, log SHA256, and write a `.size` sidecar to detect cache corruption. Full SHA256 verification against Hub metadata is not yet wired (Hub listing does not expose hashes); compare the logged hash with the Hub file page when debugging.
- Convert buffers tensors in memory (`GGUFWriter` + `recast` peak ~2x model size). A 2.5B F32 model needs ~10 GB RAM. Streaming/mmap conversion is future work; GGUF headers are capped (`MAX_KV_COUNT`, `MAX_TENSOR_COUNT`, `MAX_STRING_LEN`) so corrupt files fail fast instead of OOMing.
- Status goes to stderr, JSON to stdout. Pass `zsfm --quiet` (or `ZSFM_QUIET=1`) to silence status for scripts.

## Releasing

Pushing a `v*` tag triggers `.github/workflows/release.yml`: cross-platform binary builds (`cargo build --release -p zsfm`), a GitHub Release with `--generate-notes`, and (gated behind the `PUBLISH_CRATES_IO` repository variable) publishing all 24 crates to crates.io in dependency order via `zsfm-rs/scripts/publish-crates.sh` (`cargo publish -p zsfm` is last).
