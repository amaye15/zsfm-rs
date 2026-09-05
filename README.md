# zsfm-rs — Zero-Shot Forecasters & Tabular Foundation Models in Rust

> **One `zsfm` binary, 16 foundation models, no Python.** Download real weights from Hugging Face, convert to a single self-contained [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file, and run zero-shot inference locally via [candle](https://github.com/huggingface/candle).

[![CI](https://github.com/amaye15/zsfm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/amaye15/zsfm-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

**Repo:** `amaye15/zsfm-rs` (renamed from `zero-shot-forecasters-gguf`) · **Binary:** `zsfm` · **Workspace:** `zsfm-rs/`

---

## What is this?

`zsfm-rs` is a Rust workspace that ports **11 zero-shot time-series forecasters** and **5 tabular foundation models** to a unified GGUF + candle stack. Each port is verified **bit-exact** (or within F32 rounding, ~1e-6–1e-7) tensor-for-tensor against the original PyTorch checkpoint before being considered done — no re-training, no approximations, just the original published weights re-encoded.

There are two halves:

1. **Convert** — `zsfm <model> convert` downloads the model's real weights from Hugging Face and writes a single `*.gguf` file.
2. **Infer** — `zsfm <model> infer --gguf <file> < request.json` loads that GGUF and returns a forecast / tabular prediction. No PyTorch, no Python runtime.

The same GGUF file works with any GGUF consumer; `infer` just happens to use candle.

**Docs:** [User Guide (mdBook)](https://amaye15.github.io/zsfm-rs/guide/) · [API docs (rustdoc)](https://amaye15.github.io/zsfm-rs/api/) · [Benchmark](benchmark.md)

---

## Supported models

### Time-series forecasters (11)

All read a numeric `context` (past values) + `horizon` (steps to predict) and return an OpenAI-compatible `{ point, quantiles }` forecast. Toto / Moirai / Moirai-2 also support batch and multivariate input.

| Model | HF weights | Original code | Paper | License | Output |
|-------|------------|---------------|-------|---------|--------|
| **Toto-2** (Datadog) | [Datadog/Toto-2.0-2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B) | [DataDog/toto](https://github.com/DataDog/toto) | [Toto 2.0](https://arxiv.org/abs/2605.20119) | Apache-2.0 | quantiles |
| **Chronos-2** (Amazon) | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | [Chronos-2](https://arxiv.org/abs/2510.15821) | Apache-2.0 | quantiles |
| **TimesFM 2.5** (Google) | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | [google-research/timesfm](https://github.com/google-research/timesfm) | [TimesFM](https://arxiv.org/abs/2310.10688) | Apache-2.0 | quantiles |
| **Sundial** (thuml) | [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m) | [thuml/Sundial](https://github.com/thuml/Sundial) | [Sundial](https://arxiv.org/abs/2502.00816) | Apache-2.0 | point only |
| **TTM** (IBM Granite) | [ibm-granite/granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm) | [TTM](https://arxiv.org/abs/2401.03955) | Apache-2.0 | point only |
| **Lag-Llama** | [time-series-foundation-models/Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama) | [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama) | [Lag-Llama](https://arxiv.org/abs/2310.08278) | Apache-2.0 | point only |
| **MOMENT** (CMU) | [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) | [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) | [MOMENT](https://arxiv.org/abs/2402.03885) | MIT | point only |
| **Moirai 1.0** (Salesforce) | [Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | [Moirai](https://arxiv.org/abs/2402.02592) | CC-BY-NC-4.0 ⚠️ | point only |
| **Moirai 2.0** (Salesforce) | [Salesforce/moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | [Moirai 2.0](https://arxiv.org/abs/2511.11698) | CC-BY-NC-4.0 ⚠️ | point only |
| **FlowState-R1** (IBM) | [ibm-granite/granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1) | [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/flowstate) | [FlowState](https://arxiv.org/abs/2508.05287) | Apache-2.0 | quantiles |
| **TiRex** (NX-AI) | [NX-AI/TiRex](https://huggingface.co/NX-AI/TiRex) | [NX-AI/tirex](https://github.com/NX-AI/tirex) | [TiRex](https://arxiv.org/abs/2505.23719) | NXAI Community ⚠️ | quantiles |

### Tabular foundation models (5)

Zero-shot in-context classification/regression: give a small labeled support set + query rows, get predictions in one forward pass — no fine-tuning.

| Model | HF weights | Original code | Paper | License | Class. | Regr. |
|-------|------------|---------------|-------|---------|--------|-------|
| **Mitra** (AutoGluon) | [autogluon/mitra-classifier](https://huggingface.co/autogluon/mitra-classifier) / [mitra-regressor](https://huggingface.co/autogluon/mitra-regressor) | [autogluon/autogluon](https://github.com/autogluon/autogluon/tree/master/tabular/src/autogluon/tabular/models/mitra) | [Mitra](https://arxiv.org/abs/2510.21204) | Apache-2.0 | ✅ | ✅ |
| **TabDPT** (Layer6) | [Layer6/TabDPT](https://huggingface.co/Layer6/TabDPT) | [layer6ai-labs/TabDPT-inference](https://github.com/layer6ai-labs/TabDPT-inference) | [TabDPT](https://huggingface.co/papers/2410.18164) | Apache-2.0 | ✅ | ✅ |
| **TabICL v2** | [jingang/TabICL](https://huggingface.co/jingang/TabICL) | [soda-inria/tabicl](https://github.com/soda-inria/tabicl) | [TabICL](https://arxiv.org/abs/2502.05564) / [v2](https://arxiv.org/abs/2602.11139) | BSD-3-Clause | ✅ | ❌ |
| **TabPFN-3** (Prior Labs) | [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | [TabPFN-3](https://arxiv.org/abs/2605.13986) | Non-commercial ⚠️ | ✅ | ❌ |
| **TabFM** (Google) | [google/tabfm-1.0.0-pytorch](https://huggingface.co/google/tabfm-1.0.0-pytorch) | [google-research/tabfm](https://github.com/google-research/tabfm) | [Blog post](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) | Non-commercial ⚠️ | ✅ | ✅ |

> ⚠️  **Moirai / TabPFN-3 / TabFM are non-commercial only** (research & internal evaluation). **TiRex** is free commercially unless your org's consolidated annual revenue exceeds €100M and you ship TiRex in a commercial product/service. See [Licensing](#licensing) and [docs-guide/src/licensing.md](docs-guide/src/licensing.md).

---

## Installation

### Option 1: `cargo install` (like `cargo install ripgrep`)

From [crates.io](https://crates.io/crates/zsfm) (requires a recent stable Rust toolchain from [rustup.rs](https://rustup.rs)):

```bash
cargo install zsfm --locked
zsfm --help
```

From git (latest `main`):

```bash
cargo install --git https://github.com/amaye15/zsfm-rs zsfm --locked
# or pin a tag:
cargo install --git https://github.com/amaye15/zsfm-rs --tag v0.1.0 zsfm --locked
```

From a local checkout:

```bash
git clone https://github.com/amaye15/zsfm-rs.git
cargo install --path zsfm-rs/crates/zsfm --locked
# or, build without installing:
cargo build --release -p zsfm --manifest-path zsfm-rs/Cargo.toml
./zsfm-rs/target/release/zsfm --help
```

> `--locked` uses the `Cargo.lock` tested in CI for reproducible builds. Omit it if you want the latest compatible dependencies.

### Option 2: pre-built binary

Binaries for Linux (`x86_64-unknown-linux-gnu`) and macOS Apple Silicon (`aarch64-apple-darwin`) are published on the [Releases](https://github.com/amaye15/zsfm-rs/releases) page:

```bash
tar xzf zsfm-v0.1.0-aarch64-apple-darwin.tar.gz
sudo mv zsfm*/*zsfm /usr/local/bin/
zsfm --help
```

> macOS Intel (`x86_64-apple-darwin`) is not currently published (GitHub's free Intel runner capacity was cut). Use `cargo install` above instead — the workspace builds fine on Intel Macs.

On macOS the release profile automatically links Apple's Accelerate framework for faster BLAS; on Linux/Windows it falls back to candle's portable backend with no extra setup.

Verify:

```bash
zsfm --help              # one subcommand per model
zsfm chronos --help      # per-model: convert / infer / upload / inspect-tensors
```

---

## Quick start

### Time-series forecasting (Moirai-2, ~45 MB)

```bash
# 1. Download + convert to GGUF (cached at models/Salesforce__moirai-2.0-R-small/model-f32.gguf)
zsfm moirai2 convert

# 2. Build a request (Moirai batches series; univariate models take a flat array)
cat > request.json <<'EOF'
{
  "context": [[10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.3, 11.8, 12.0, 12.2]],
  "horizon": 4
}
EOF

# 3. Forecast
zsfm moirai2 infer --gguf gguf/moirai2-f32.gguf < request.json
```

Response (OpenAI-compatible):

```json
{
  "id": "forecast-...",
  "object": "forecast",
  "model": "moirai-2",
  "choices": [{ "index": 0, "forecast": { "point": [12.35, 12.51, 12.64, 12.72] } }]
}
```

Univariate models use a flat array instead:

```bash
zsfm chronos convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm chronos infer --gguf gguf/chronos-f16.gguf
```

### Tabular prediction (Mitra)

```bash
zsfm mitra convert --model autogluon/mitra-classifier --task classification

cat > request.json <<'EOF'
{
  "x_support": [[0.1, 1.2], [0.9, -0.3], [-1.1, 0.4], [1.5, 1.1]],
  "y_support": [0, 1, 1, 0],
  "x_query": [[0.2, 0.9], [-0.8, 0.1]],
  "n_classes": 2
}
EOF

zsfm mitra infer --gguf gguf/mitra-classification-f32.gguf < request.json
# {"task":"classification","probabilities":[[0.83,0.17],[0.21,0.79]]}
```

See [Time-series forecasters](docs-guide/src/models/time-series.md) and [Tabular foundation models](docs-guide/src/models/tabular.md) for every model's exact request/response shape, flags, and caveats.

---

## CLI overview

Every model is a subcommand: `zsfm <model> <action>`.

| Action | What it does |
|--------|--------------|
| `convert` | Download from Hugging Face and write a GGUF file |
| `infer` | Load a GGUF and run inference on JSON from stdin → JSON to stdout |
| `upload` | Push source + GGUF to a Hugging Face repo you control |
| `inspect-tensors` | Print every tensor name/shape/dtype in a local checkpoint |

```bash
zsfm <model> --help
zsfm <model> convert --help
zsfm <model> infer --help
```

#### Model caching

The first `convert` for a model:

1. Downloads raw weights from HF.
2. Writes a **canonical F32 GGUF** to `<model-dir>/<owner>__<repo>/model-f32.gguf` (default `model-dir` is `models/`).
3. Deletes the raw weight files (they're redundant once the canonical GGUF exists).

Later converts for the same model — at any `--dtype` — recast from that cache with no network:

```bash
zsfm chronos convert --dtype f32   # downloads
zsfm chronos convert --dtype f16   # <1s, from cache
zsfm chronos convert --dtype q8    # <1s, from cache (Q8_0 block-quant; tiny tensors stay F32)
zsfm chronos convert --redownload  # force a fresh download
```

Most `convert` commands accept `--dtype f32|f16|q8`. BF16 is available only through the generic converter (candle can't currently load GGUF BF16 back).

#### Generic converter

TabICL and TabPFN-3 ship checkpoints whose tensor names are used unchanged, so they go through a model-agnostic path (also works for any HF repo):

```bash
zsfm convert --repo jingang/TabICL --file classifier-v2 --format ckpt -o gguf/tabicl-v2-f32.gguf
zsfm convert --repo Prior-Labs/tabpfn_3 --file classifier-v3_default --format ckpt -o gguf/tabpfn-v3-f32.gguf
# re-quantize an existing GGUF:
zsfm convert existing.gguf -o smaller.gguf --dtype q8
```

`--file` is a case-insensitive substring to disambiguate repos that publish multiple checkpoints.

---

## Benchmark

`zsfm-bench` (`zsfm-rs/crates/zsfm-bench/`) links all 11 forecasters in-process via the shared `zsfm_core::Forecaster` trait — each GGUF is loaded once and independent models run concurrently.

```bash
# convert the models you want to evaluate first
zsfm ttm convert && zsfm moirai2 convert

# single dataset
zsfm-bench run --dataset ETTh1 --models ttm,moirai2 --windows 30

# full sweep (21 datasets × models × horizons/contexts) → writes ../benchmark.md
zsfm-bench report
```

Results are cached in `benchmark/bench_cache.json` (gitignored) so re-runs and interruptions only compute what's missing. The full pre-computed results live in [benchmark.md](benchmark.md):

- **21 datasets** across Energy, Climate, Astronomy, Health, Finance, Transport, Hydrology — from ETTh/m (Transformer benchmark) through Monash and Informer collections to Jena/Saugeen.
- **30 rolling windows** at `context=512, horizon=96` for the main table, plus horizon sweeps (`96/192/336/720`) and context sweeps (`96/256/512/1024`).
- **11 ensemble strategies** evaluated over all non-trivial model subsets (mean, median, inverse-MAE weighted, trimmed mean, softmax-weighted, geometric mean, online adaptive, Hedge, EMA-smoothed, greedy selection, per-horizon and uncertainty-weighted).
- **Metrics:** MAE, RMSE, MASE (scale-free vs. naïve 1-step baseline), and per-window inference latency.

Highlights from `benchmark.md:140-167` (best ensemble vs. best solo, `context=512, horizon=96, 30 windows`): ensembles beat the best single model on 19/21 datasets (up to **−9.7%** on Wind), with the median-based strategy `(~)` winning most often; only `aus_electricity` and `weather` favor a solo model. The appendix in `benchmark.md:212-952` has per-dataset Top-10 ensembles.

See also [benchmark/optimisation_log.md](benchmark/optimisation_log.md) for inference latency optimisation history (LTO, Accelerate, etc.).

---

## Project structure

```
.
├── Cargo.toml                # root workspace (so `cargo install --git https://github.com/amaye15/zsfm-rs zsfm` works)
├── zsfm-rs/                  # Rust workspace (all code lives here; also usable as `cargo build -p zsfm --manifest-path zsfm-rs/Cargo.toml`)
│   ├── Cargo.toml            # workspace + shared [profile.release] (lto=fat, opt-level=3, strip)
│   ├── crates/
│   │   ├── zsfm/             # `zsfm` binary — one subcommand module per model (`cargo install zsfm`)
│   │   ├── zsfm-gguf/        # GGUF reader/writer
│   │   ├── zsfm-hub/         # HF download/upload + canonical-F32 cache helpers
│   │   ├── zsfm-checkpoint/  # checkpoint loader (safetensors/pickle/ONNX/HDF5/npz/ckpt/gguf) + dtype casting
│   │   ├── zsfm-core/        # `Forecaster` trait, JSON envelope, config helpers
│   │   ├── zsfm-nn/          # shared candle tensor ops
│   │   ├── zsfm-bench/       # in-process benchmark + ensembling
│   │   └── models/           # one crate per model (arch + weight mapping + candle kernels)
│   │       ├── toto/ chronos/ timesfm/ sundial/ ttm/ lag_llama/ moment/
│   │       ├── moirai/ moirai2/ flowstate/ tirex/   # forecasters
│   │       └── mitra/ tabdpt/ tabicl/ tabpfn/ tabfm/ # tabular
│   ├── scripts/publish-crates.sh
│   └── tools/export-weights.py
├── docs-guide/               # mdBook user guide (published to GitHub Pages /guide/)
│   └── src/
│       ├── introduction.md
│       ├── installation.md
│       ├── quickstart.md
│       ├── cli-overview.md
│       ├── models/time-series.md
│       ├── models/tabular.md
│       ├── licensing.md
│       └── development.md
├── benchmark/
│   ├── data/                 # evaluation datasets (not all tracked; see .gitignore)
│   └── optimisation_log.md
├── .github/
│   ├── workflows/ci.yml      # cargo build + cargo test (Linux + macOS)
│   ├── workflows/release.yml # cross-platform binaries + crates.io publish (on v* tag)
│   └── workflows/docs.yml    # mdBook + rustdoc → GitHub Pages
│   └── pages/                # custom landing page (guide + API cards)
├── benchmark.md              # pre-computed benchmark report (21 datasets, sweeps, ensembles)
└── LICENSE (MIT)
```

Each model crate under `zsfm-rs/crates/models/` owns its architecture, checkpoint weight-name mapping, and candle kernels; `zsfm` (`zsfm-rs/crates/zsfm/`) only wires them to the CLI.

---

## Development

```bash
cd zsfm-rs
cargo build --release --workspace
cargo test  --release --workspace   # ~130 tests (GGUF round-trip, dtype casting, arch shapes, bench logic)
```

CI (`.github/workflows/ci.yml:1`) runs both on every push to `main` and every PR (Linux + macOS).

### Adding a new model

1. New crate at `zsfm-rs/crates/models/<name>/` (weight mapping + candle inference + request/response types).
2. Module at `zsfm/src/<name>.rs` (`zsfm-rs/crates/zsfm/src/<name>.rs`) with `convert`/`infer` subcommands — follow the caching pattern in [The `zsfm` CLI](docs-guide/src/cli-overview.md#model-caching) (`zsfm_hub::canonical_gguf_path` + `zsfm_checkpoint::recast`).
3. Wire it into `zsfm/src/main.rs:ModelCommand` (`zsfm-rs/crates/zsfm/src/main.rs`).
4. Add a page under `docs-guide/src/models/` and a row in the model table.

See [Development](docs-guide/src/development.md) for benchmarking and release notes. Pushing a `v*` tag triggers `.github/workflows/release.yml:1` (cross-platform binaries via `cargo build -p zsfm` + GitHub Release + optional crates.io publish via `zsfm-rs/scripts/publish-crates.sh` — `cargo publish -p zsfm` is last).

---

## Documentation

- **User guide (mdBook):** `docs-guide/` → https://amaye15.github.io/zsfm-rs/guide/ — installation, quick start, CLI overview, per-model pages with exact JSON shapes and links to papers/code/weights.
- **API reference (rustdoc):** `cargo doc` for every crate → https://amaye15.github.io/zsfm-rs/api/
- **Landing page:** `.github/pages/index.html` — cards linking to guide + per-crate rustdoc.

Build the guide locally:

```bash
cargo install mdbook
mdbook serve docs-guide
```

---

## Licensing

**Code** in this repository (converters, GGUF writer/reader, CLI, candle kernels, bench) is **MIT** — see [LICENSE](LICENSE).

**Model weights** keep the license their original authors published. Converting with `zsfm <model> convert` downloads the original weights and re-encodes them as GGUF — it does not change ownership or terms. Check before using a GGUF beyond local experimentation:

| License | Models | Commercial use |
|---------|--------|----------------|
| Apache-2.0 | Toto-2, Chronos-2, TimesFM 2.5, Sundial, TTM, Lag-Llama, Mitra, TabDPT, FlowState-R1 | ✅ unrestricted |
| MIT | MOMENT | ✅ unrestricted |
| BSD-3-Clause | TabICL | ✅ unrestricted |
| NXAI Community | TiRex | ✅ unless your org's annual revenue > €100M *and* you ship TiRex in a commercial product/service |
| CC-BY-NC-4.0 | Moirai 1.0, Moirai 2.0 | ❌ non-commercial only |
| tabpfn-3-license-v1.0 | TabPFN-3 | ❌ non-commercial only |
| TabFM Non-Commercial v1.0 | TabFM | ❌ non-commercial only |

For the restricted models, see the license file in each HF repo (linked in the tables above) and [docs-guide/src/licensing.md](docs-guide/src/licensing.md). This project does not interpret or enforce weight licenses.

---

## Acknowledgments

This workspace is a packaging and portability layer — all modeling credit belongs to the original authors. Each crate and each page in [Time-series forecasters](docs-guide/src/models/time-series.md) / [Tabular foundation models](docs-guide/src/models/tabular.md) links back to the original Hugging Face weights, source repository, and paper. If you use these models in academic work, please cite the relevant paper(s) from those pages.

Built on [candle](https://github.com/huggingface/candle) (Hugging Face) and the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format (ggml/ggerganov).

---

*Maintained at [amaye15/zsfm-rs](https://github.com/amaye15/zsfm-rs). Issues and PRs welcome — see [Development](docs-guide/src/development.md). Feedback on the tooling itself: https://github.com/anomalyco/opencode.*
