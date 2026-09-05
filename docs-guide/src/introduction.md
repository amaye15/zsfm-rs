# Introduction

`zsfm-rs` is a Rust workspace for running zero-shot forecasting and tabular foundation models locally, without Python. It has two halves — plus a cleanup helper:

1. **Convert** — download a model's real weights from HuggingFace and turn them into a single self-contained [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) file.
2. **Infer** — load that GGUF file and run zero-shot inference (a forecast, or a tabular classification/regression) via [candle](https://github.com/huggingface/candle), Hugging Face's Rust tensor library. No PyTorch, no Python runtime, one native binary.
3. **Delete** — remove a model's cached `model-f32.gguf` + `config.json` (and optionally an output GGUF) to free disk (`zsfm chronos delete`).

Everything is exposed through one CLI binary, `zsfm`, with a subcommand per model:

```bash
zsfm chronos convert          # download + convert Chronos-2 to GGUF
zsfm chronos infer --gguf ... # run a forecast
zsfm chronos delete           # remove the cached files for Chronos-2
```

## What's included

**11 time-series forecasters** — point or quantile forecasts from a numeric context window:

Toto-2, Chronos-2, TimesFM 2.5, Sundial, TTM, Lag-Llama, MOMENT, Moirai 1.0, Moirai 2.0, FlowState-R1, TiRex.

**5 tabular foundation models** — zero-shot classification/regression from a small labeled support set, no fine-tuning:

Mitra, TabDPT, TabICL, TabPFN-3, TabFM.

See [Time-series forecasters](./models/time-series.md) and [Tabular foundation models](./models/tabular.md) for the exact request/response JSON for each.

## Correctness

Every model in this workspace was verified **bit-exact** (or within pure F32 rounding, typically ~1e-6 to ~1e-7) against a reference Python implementation running the real downloaded checkpoint, before being considered done. This isn't a from-scratch reimplementation guessing at architecture — each port was checked tensor-by-tensor against the original. Every model's page links back to its original HuggingFace weights, the original authors' source repo, and the paper it came from — this project converts and runs those exact weights, it doesn't retrain or approximate them.

## Where to go next

- New to the project? Start with [Installation](./installation.md) and [Quick start](./quickstart.md).
- Want the exact JSON shapes for a specific model, or a link to its original repo/paper? Jump straight to [Time-series forecasters](./models/time-series.md) or [Tabular foundation models](./models/tabular.md).
- Three of the 16 models (Moirai, TabPFN-3, TabFM) are **non-commercial only** — see [Licensing](./licensing.md) before using them beyond research/internal evaluation.
- Looking for the generated Rust API reference (types, function signatures) rather than a usage guide? See the [API docs](../api/index.html).
