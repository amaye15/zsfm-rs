---
license: mit
library_name: gguf
tags:
  - gguf
  - time-series
  - forecasting
  - rust
base_model: google/timesfm-2.5-200m-pytorch
---

# timesfm-rs

Pure Rust converter and inference engine for [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch).

Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/timesfm-rs convert --model google/timesfm-2.5-200m-pytorch --dtype f16 --output gguf/timesfm-f16.gguf

# Q8_0 (smallest)
./target/release/timesfm-rs convert --dtype q8 --output gguf/timesfm-q8.gguf

# F32 (full precision)
./target/release/timesfm-rs convert --dtype f32 --output gguf/timesfm-f32.gguf
```

To convert all dtypes at once:

```bash
./scripts/convert_all.sh
```

HuggingFace token (optional for public models):

```bash
HF_TOKEN=hf_... ./scripts/convert_all.sh
```

## Inspect tensors

Print all tensor names and shapes from a GGUF file:

```bash
./target/release/timesfm-rs inspect-tensors gguf/timesfm-f16.gguf
```

## Infer

Run forecasting from comma-separated context values:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 64}' \
  | ./target/release/timesfm-rs infer --gguf gguf/timesfm-f16.gguf
```

Output is JSON in an OpenAI-compatible forecast format with point forecast and all 9 quantile channels (q0.1–q0.9):

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "timesfm",
  "choices": [{
    "index": 0,
    "forecast": {
      "point": [2.1, 2.3, 2.5, "..."],
      "quantiles": {
        "0.1": [1.8, 2.0, 2.2, "..."],
        "0.5": [2.1, 2.3, 2.5, "..."],
        "0.9": [2.4, 2.6, 2.8, "..."]
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {"context_length": 8, "forecast_length": 64}
}
```

**Batch inference** — pass multiple series as a nested array to get one `Choice` per series:

```bash
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 64}' \
  | ./target/release/timesfm-rs infer --gguf gguf/timesfm-f16.gguf
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import timesfm_rs

model = timesfm_rs.TimesFM("gguf/timesfm-f16.gguf")

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

TimesFM 2.5 is a decoder-only patch transformer:

- **Input**: Context is split into 32-value patches; each patch is instance-normalized (RevIN with cumulative running statistics) and concatenated with an observation mask, producing 64-dim tokenizer inputs
- **Backbone**: 20-layer causal transformer, d_model=1280, 16 heads, head_dim=80, d_ff=1280; 200M parameters
- **Attention**: Sandwich norms (pre-norm → attention → post-norm + residual); fused QKV projection; per-dimension query scale `1.442695/√80 × softplus(param)`; QK RMSNorm after RoPE
- **AR decoding**: KV cache with 4-patch decode stride — O(n) instead of O(n²) re-forward per step
- **Output**: Each output patch decoded through a ResidualBlock to `[n_patches, 128, 10]`; index 0 is point forecast, indices 1–9 are quantiles q0.1–q0.9
