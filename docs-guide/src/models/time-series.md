# Time-series forecasters

All 11 models here read a numeric `context` (past values) and a `horizon` (how many future steps to predict), and write back an [OpenAI-compatible](https://platform.openai.com/docs/) forecast object. `zsfm <model> infer --help` always shows the exact request shape for that model.

Two request shapes are used, depending on the model:

- **Univariate**: `{"context": [...], "horizon": N}` — one flat array of numbers.
- **Batch** (Toto, Moirai, Moirai-2): `{"context": [[...], [...]], "horizon": N}` — a list of series, forecast independently. These three also accept a multivariate form (`[[[v0_t0, ...], [v1_t0, ...]], ...]`) for genuinely multi-channel input.

Response shape is always:

```json
{
  "id": "forecast-...",
  "object": "forecast",
  "model": "<model-name>",
  "choices": [
    {"index": 0, "forecast": {"point": [...], "quantiles": {...}}}
  ]
}
```

Not every model produces quantiles — point-forecast-only models (noted below) only fill in `"point"`.

Each row below is the original model, not a reimplementation with a different architecture — `zsfm convert` downloads the exact published weights and this workspace's inference code is verified bit-exact (or numerically equivalent within float tolerance) against the original PyTorch implementation. Links go to the original HuggingFace weights, the original authors' source repo, and the paper.

| Model | HF weights | Original code | License | Output |
|---|---|---|---|---|
| [Toto-2](#toto-2) | [Datadog/Toto-2.0-2.5B](https://huggingface.co/Datadog/Toto-2.0-2.5B) | [DataDog/toto](https://github.com/DataDog/toto) | Apache-2.0 | quantiles |
| [Chronos-2](#chronos-2) | [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Apache-2.0 | quantiles |
| [TimesFM 2.5](#timesfm-25) | [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | [google-research/timesfm](https://github.com/google-research/timesfm) | Apache-2.0 | quantiles |
| [Sundial](#sundial) | [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m) | [thuml/Sundial](https://github.com/thuml/Sundial) | Apache-2.0 | point only |
| [TTM](#ttm) | [ibm-granite/granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm) | Apache-2.0 | point only |
| [Lag-Llama](#lag-llama) | [time-series-foundation-models/Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama) | [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama) | Apache-2.0 | point only |
| [MOMENT](#moment) | [AutonLab/MOMENT-1-large](https://huggingface.co/AutonLab/MOMENT-1-large) | [moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) | MIT | point only |
| [Moirai 1.0](#moirai-10--moirai-20) | [Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | **CC-BY-NC-4.0** | point only |
| [Moirai 2.0](#moirai-10--moirai-20) | [Salesforce/moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | **CC-BY-NC-4.0** | point only |
| [FlowState-R1](#flowstate-r1) | [ibm-granite/granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1) | [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/flowstate) | Apache-2.0 | quantiles |
| [TiRex](#tirex) | [NX-AI/TiRex](https://huggingface.co/NX-AI/TiRex) | [NX-AI/tirex](https://github.com/NX-AI/tirex) | **NXAI Community** | quantiles |

Bold licenses have real usage restrictions beyond plain permissive — see [Licensing](../licensing.md) before relying on Moirai or TiRex weights for anything beyond research/internal use.

## Toto-2

Paper: [Toto 2.0: Time Series Forecasting Enters the Scaling Era](https://arxiv.org/abs/2605.20119).

```bash
zsfm toto convert
echo '{"context": [[1,2,3,4,5,6,7,8]], "horizon": 4}' | zsfm toto infer --gguf gguf/toto-2.5b-f16.gguf
```

Batch and multivariate input supported (see table above). `--context-length` on `infer` overrides how much of the context window is fed to the model (default: last 4096 steps, must be divisible by the patch size, 32). `--f64` runs the forward pass in double precision to match PyTorch's numerical accuracy more closely, at ~2x memory.

## Chronos-2

Paper: [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821).

```bash
zsfm chronos convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm chronos infer --gguf gguf/chronos-f16.gguf
```

Univariate only. Full quantile levels in the response; `"point"` is the median (q0.5).

## TimesFM 2.5

Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688) (ICML 2024).

```bash
zsfm timesfm convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm timesfm infer --gguf gguf/timesfm.gguf
```

Univariate only. Fixed architecture — no `--config` flag needed at inference time (everything's embedded in the GGUF).

## Sundial

Paper: [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816) (ICML 2025 Oral).

```bash
zsfm sundial convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm sundial infer --gguf gguf/sundial-f16.gguf
```

Flow-matching model, point-forecast only. `--steps` on `infer` overrides the ODE solver's step count (default: from GGUF metadata, typically 50); 10-20 is usually enough and latency scales linearly with this value.

## TTM

Paper: [TinyTimeMixers](https://arxiv.org/abs/2401.03955) (NeurIPS 2024).

```bash
zsfm ttm convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm ttm infer --gguf gguf/ttm-f32.gguf
```

Univariate, point-forecast only. Small and fast (~3MB as F32).

## Lag-Llama

Paper: [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278).

```bash
zsfm lag-llama convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm lag-llama infer --gguf gguf/lag_llama-f32.gguf
```

Univariate, point-forecast only. Downloads a raw PyTorch Lightning `.ckpt` and reads it directly — no Python needed for conversion.

## MOMENT

Paper: [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885) (ICML 2024).

```bash
zsfm moment convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm moment infer --gguf gguf/moment-f32.gguf
```

Univariate, point-forecast only.

## Moirai 1.0 / Moirai 2.0

Papers: [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592) (Moirai 1.0), [Moirai 2.0: When Less Is More for Time Series Forecasting](https://arxiv.org/abs/2511.11698) (Moirai 2.0).

```bash
zsfm moirai convert   # or: zsfm moirai2 convert
echo '{"context": [[1,2,3,4,5,6,7,8]], "horizon": 4}' | zsfm moirai infer --gguf gguf/moirai-f32.gguf
```

Point-forecast only, channel-independent across variates (each variate forecast independently, computed in parallel via `rayon`). Batch and multivariate input supported. Moirai-2 is the newer, smaller (`R-small`) checkpoint.

> Both checkpoints are **CC-BY-NC-4.0** — non-commercial use only. See [Licensing](../licensing.md).

## FlowState-R1

Paper: [FlowState: Sampling Rate Invariant Time Series Forecasting](https://arxiv.org/abs/2508.05287).

```bash
zsfm flowstate convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm flowstate infer --gguf gguf/flowstate-r1-f16.gguf
```

Univariate. Full quantile levels; `"point"` is the median.

## TiRex

Paper: [TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](https://arxiv.org/abs/2505.23719). Built on [xLSTM](https://github.com/NX-AI/xlstm).

```bash
zsfm tirex convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm tirex infer --gguf gguf/tirex-f32.gguf
```

Univariate. Full quantile levels; `"point"` is the median. Downloads a raw `.ckpt` directly, like Lag-Llama.

> Licensed under the **NXAI Community License** (modeled on Meta's Llama community license): free to use and redistribute, including commercially, unless your organization's consolidated annual revenue exceeds €100M *and* you're incorporating TiRex into a commercial product or service — in which case NXAI requires a separate commercial license. See [Licensing](../licensing.md).
