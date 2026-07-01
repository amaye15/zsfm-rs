---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model:
  - Datadog/Toto-2.0-4m
  - Datadog/Toto-2.0-22m
  - Datadog/Toto-2.0-313m
  - Datadog/Toto-2.0-1B
  - Datadog/Toto-2.0-2.5B
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - probabilistic
  - transformer
  - rust
inference: false
---

# toto-rs

Pure Rust converter and inference engine for [Datadog Toto-2.0](https://huggingface.co/collections/Datadog/toto-20-6768ce5e1b4c1d0e7c5b9fed).

Pre-converted GGUF files are available at [amaye15/toto-gguf](https://huggingface.co/amaye15/toto-gguf). Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended — ~2× compression vs F32, ~0.07 MAE on models ≥313m)
./target/release/toto-rs convert --model Datadog/Toto-2.0-2.5B --dtype f16 --output gguf/toto-2.5b-f16.gguf

# Q8_0 (smallest — useful when memory is the bottleneck, model ≥313m)
./target/release/toto-rs convert --model Datadog/Toto-2.0-2.5B --dtype q8 --output gguf/toto-2.5b-q8.gguf

# F32 (full precision)
./target/release/toto-rs convert --model Datadog/Toto-2.0-2.5B --dtype f32 --output gguf/toto-2.5b-f32.gguf
```

Available models: `Datadog/Toto-2.0-{4m,22m,313m,1B,2.5B}`

To convert all five sizes in all three dtypes at once:

```bash
./scripts/convert_all.sh
```

HuggingFace token (optional for public models):

```bash
HF_TOKEN=hf_... ./scripts/convert_all.sh
```

## Inspect tensors

Print all tensor names and shapes from a `.safetensors` checkpoint:

```bash
./target/release/toto-rs inspect-tensors models/model.safetensors
```

## Infer

Run univariate quantile forecasting from stdin JSON:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 64}' \
  | ./target/release/toto-rs infer \
      --gguf gguf/toto-2.5b-f16.gguf \
      --config models/toto-2.5b/config.json
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "toto",
  "choices": [{
    "index": 0,
    "forecast": {
      "point": [2.1, 2.3, 2.5, "..."],
      "quantiles": {
        "0.10": [1.8, 2.0, 2.2, "..."],
        "0.50": [2.1, 2.3, 2.5, "..."],
        "0.90": [2.4, 2.6, 2.8, "..."]
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {"context_length": 8, "forecast_length": 64}
}
```

`point` is the median (q0.5) forecast; all 9 quantiles (q0.10–q0.90) are included.

**Batch inference** — pass multiple series as a nested array to get one `Choice` per series:

```bash
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 64}' \
  | ./target/release/toto-rs infer \
      --gguf gguf/toto-2.5b-f16.gguf \
      --config models/toto-2.5b/config.json
```

**Multivariate inference** — Toto's variate-aware architecture natively handles multiple co-occurring time series. Pass a 3D context array `[batch][variate][time]` to get a `variates` array in each choice:

```bash
echo '{
  "context": [
    [[1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1],
     [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
  ],
  "horizon": 64
}' \
  | ./target/release/toto-rs infer \
      --gguf gguf/toto-2.5b-f16.gguf \
      --config models/toto-2.5b/config.json
```

Multivariate output (one `variates` entry per variate):

```json
{
  "choices": [{
    "index": 0,
    "forecast": {
      "variates": [
        {
          "point": [2.1, 2.3, "..."],
          "quantiles": {"0.10": [...], "0.50": [...], "0.90": [...]}
        },
        {
          "point": [1.3, 1.4, "..."],
          "quantiles": {"0.10": [...], "0.50": [...], "0.90": [...]}
        }
      ]
    },
    "finish_reason": "stop"
  }]
}
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import toto_rs

model = toto_rs.Toto("gguf/toto-2.5b-f16.gguf", "models/toto-2.5b/config.json")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=64)
fc     = result["choices"][0]["forecast"]
point  = fc["point"]           # median forecast
q10    = fc["quantiles"]["0.10"]  # 10th-percentile
q90    = fc["quantiles"]["0.90"]  # 90th-percentile

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=64)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

Toto-2.0 is an encoder-decoder time series foundation model:

- **Input**: Multivariate time series; context is instance-normalized, patched (patch_size=32), and projected to d_model
- **Encoder**: Multi-layer causal transformer with xPos RoPE (extends standard RoPE with per-dimension exponential decay for long-range stability)
- **Output**: Last hidden states decoded through a ResidualBlock to 9-quantile forecasts (q0.1–q0.9) for each variate
- **Models**: Five sizes — 4m, 22m, 313m, 1B, 2.5B — all using the same architecture

## Benchmark

Measured on Apple M-series (CPU), 512-step context → 64-step forecast, 3 runs averaged.

| Model     | dtype |  Size (GB) | Time (s) | MAE vs F32 |
|-----------|-------|-----------:|---------:|-----------:|
| toto-4m   | f32   |       0.02 |     0.01 |          — |
|           | f16   |       0.01 |     0.02 |      0.176 |
|           | q8    |       0.00 |     0.01 |      3.242 |
| toto-22m  | f32   |       0.09 |     0.04 |          — |
|           | f16   |       0.04 |     0.04 |      0.071 |
|           | q8    |       0.02 |     0.04 |      5.030 |
| toto-313m | f32   |       1.25 |     0.32 |          — |
|           | f16   |       0.63 |     0.36 |      0.074 |
|           | q8    |       0.33 |     0.39 |      0.316 |
| toto-1b   | f32   |       4.16 |     1.20 |          — |
|           | f16   |       2.08 |     1.19 |      0.074 |
|           | q8    |       1.11 |     0.87 |      0.334 |
| toto-2.5b | f32   |       9.82 |     5.40 |          — |
|           | f16   |       4.91 |     3.04 |      0.066 |
|           | q8    |       2.61 |     3.49 |      0.243 |
