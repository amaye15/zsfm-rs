# Development

## Workspace layout

```
zsfm-rs/
  crates/
    zsfm-cli/        # the `zsfm` binary — one subcommand module per model
    zsfm-gguf/        # GGUF reader/writer
    zsfm-checkpoint/  # loads safetensors/pickle/onnx/hdf5/npz/ckpt/gguf, dtype casting, recast()
    zsfm-hub/         # HuggingFace download + the canonical-F32-cache helpers
    zsfm-tensor/      # shared tensor/quantization helpers
    models/
      chronos/ flowstate/ moirai/ moirai2/ moment/ sundial/ timesfm/ toto/ ttm/
      lag_llama/ tirex/                          # time-series forecasters
      mitra/ tabdpt/ tabicl/ tabpfn/ tabfm/       # tabular foundation models
```

Each model crate under `models/` owns its architecture, weight-name mapping, and inference kernel; `zsfm-cli` just wires them up to `convert`/`infer`/`upload`/`inspect-tensors` subcommands.

## Building and testing

```bash
cd zsfm-rs
cargo build --release --workspace
cargo test --release --workspace
```

The baseline is 115 passing tests across the workspace (unit tests for tensor casting, GGUF round-tripping, and per-model architecture/shape checks). CI (`.github/workflows/ci.yml`) runs both commands on every push to `main` and every PR, on Linux and macOS.

## Verifying a model port is correct

Every model in this workspace was verified **bit-exact** (or numerically equivalent within float tolerance) against its original PyTorch implementation before being considered done — same input, same output, checkpoint tensor-for-tensor. If you're modifying a model crate, re-run that model's `convert` + `infer` against a known input/output pair before assuming a change is safe; there isn't a single workspace-wide golden-output test harness, so this is a manual step per model.

## Adding a new model

Roughly the shape to follow, based on the existing crates under `models/`:

1. New crate under `crates/models/<name>/` with a weight-name mapping from the original checkpoint to whatever internal names you want, an inference module built on candle, and a request/response type.
2. A `zsfm-cli/src/<name>.rs` with `convert`/`infer` subcommands, following the caching pattern described in [The `zsfm` CLI](./cli-overview.md#model-caching) — compute `zsfm_hub::canonical_gguf_path`, check it before downloading, call `zsfm_checkpoint::recast` for cache hits and for the final requested dtype after a fresh download.
3. Wire the subcommand into `zsfm-cli`'s top-level `Commands` enum.
4. A page in this guide (`docs-guide/src/models/`) and a row in the relevant summary table.

## Releasing

Pushing a `v*` tag triggers `.github/workflows/release.yml`: cross-platform binary builds, a GitHub Release with `--generate-notes`, and (gated behind the `PUBLISH_CRATES_IO` repository variable) publishing all 22 crates to crates.io in dependency order via `zsfm-rs/scripts/publish-crates.sh`.
