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

| Model | HF repo | Default convert dtype | Output |
|---|---|---|---|
| [Toto-2](#toto-2) | `Datadog/Toto-2.0-2.5B` | f16 | quantiles |
| [Chronos-2](#chronos-2) | `amazon/chronos-2` | f16 | quantiles |
| [TimesFM 2.5](#timesfm-25) | `google/timesfm-2.5-200m-pytorch` | f16 | quantiles |
| [Sundial](#sundial) | `thuml/sundial-base-128m` | f16 | point only |
| [TTM](#ttm) | `ibm-granite/granite-timeseries-ttm-r2` | f32 | point only |
| [Lag-Llama](#lag-llama) | `time-series-foundation-models/Lag-Llama` | f32 | point only |
| [MOMENT](#moment) | `AutonLab/MOMENT-1-large` | f32 | point only |
| [Moirai 1.0](#moirai-10--moirai-20) | `Salesforce/moirai-1.0-R-large` | f32 | point only |
| [Moirai 2.0](#moirai-10--moirai-20) | `Salesforce/moirai-2.0-R-small` | f32 | point only |
| [FlowState-R1](#flowstate-r1) | `ibm-granite/granite-timeseries-flowstate-r1` | f16 | quantiles |
| [TiRex](#tirex) | `NX-AI/TiRex` | f32 | quantiles |

## Toto-2

```bash
zsfm toto convert
echo '{"context": [[1,2,3,4,5,6,7,8]], "horizon": 4}' | zsfm toto infer --gguf gguf/toto-2.5b-f16.gguf
```

Batch and multivariate input supported (see table above). `--context-length` on `infer` overrides how much of the context window is fed to the model (default: last 4096 steps, must be divisible by the patch size, 32). `--f64` runs the forward pass in double precision to match PyTorch's numerical accuracy more closely, at ~2x memory.

## Chronos-2

```bash
zsfm chronos convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm chronos infer --gguf gguf/chronos-f16.gguf
```

Univariate only. Full quantile levels in the response; `"point"` is the median (q0.5).

## TimesFM 2.5

```bash
zsfm timesfm convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm timesfm infer --gguf gguf/timesfm.gguf
```

Univariate only. Fixed architecture — no `--config` flag needed at inference time (everything's embedded in the GGUF).

## Sundial

```bash
zsfm sundial convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm sundial infer --gguf gguf/sundial-f16.gguf
```

Flow-matching model, point-forecast only. `--steps` on `infer` overrides the ODE solver's step count (default: from GGUF metadata, typically 50); 10-20 is usually enough and latency scales linearly with this value.

## TTM

```bash
zsfm ttm convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm ttm infer --gguf gguf/ttm-f32.gguf
```

Univariate, point-forecast only. Small and fast (~3MB as F32).

## Lag-Llama

```bash
zsfm lag-llama convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm lag-llama infer --gguf gguf/lag_llama-f32.gguf
```

Univariate, point-forecast only. Downloads a raw PyTorch Lightning `.ckpt` and reads it directly — no Python needed for conversion.

## MOMENT

```bash
zsfm moment convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm moment infer --gguf gguf/moment-f32.gguf
```

Univariate, point-forecast only.

## Moirai 1.0 / Moirai 2.0

```bash
zsfm moirai convert   # or: zsfm moirai2 convert
echo '{"context": [[1,2,3,4,5,6,7,8]], "horizon": 4}' | zsfm moirai infer --gguf gguf/moirai-f32.gguf
```

Point-forecast only, channel-independent across variates (each variate forecast independently, computed in parallel via `rayon`). Batch and multivariate input supported. Moirai-2 is the newer, smaller (`R-small`) checkpoint.

## FlowState-R1

```bash
zsfm flowstate convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm flowstate infer --gguf gguf/flowstate-r1-f16.gguf
```

Univariate. Full quantile levels; `"point"` is the median.

## TiRex

```bash
zsfm tirex convert
echo '{"context": [1,2,3,4,5,6,7,8], "horizon": 4}' | zsfm tirex infer --gguf gguf/tirex-f32.gguf
```

Univariate. Full quantile levels; `"point"` is the median. Downloads a raw `.ckpt` directly, like Lag-Llama.
