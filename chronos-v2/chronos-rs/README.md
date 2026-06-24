---
license: mit
library_name: gguf
tags:
  - gguf
  - time-series
  - forecasting
  - rust
base_model: amazon/chronos-2
---

# chronos-rs

Pure Rust converter and inference engine for [amazon/chronos-2](https://huggingface.co/amazon/chronos-2).

Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended — good precision/size trade-off)
./target/release/chronos-rs convert --model amazon/chronos-2 --dtype f16 --output gguf/chronos-f16.gguf

# Q8_0 (smallest, ~2.8× compression vs F16)
./target/release/chronos-rs convert --dtype q8 --output gguf/chronos-q8.gguf

# F32 (full precision)
./target/release/chronos-rs convert --dtype f32 --output gguf/chronos-f32.gguf
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

Print all tensor names and shapes from a `.safetensors` checkpoint:

```bash
./target/release/chronos-rs inspect-tensors models/model.safetensors
```

## Infer

Run univariate quantile forecasting from a GGUF file:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 64}' \
  | ./target/release/chronos-rs infer \
      --gguf gguf/chronos-f16.gguf \
      --config models/config.json
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "chronos",
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

`point` is the median (q0.5) forecast; all quantile levels from `config.json` are included.

**Batch inference** — pass multiple series as a nested array to get one `Choice` per series:

```bash
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 64}' \
  | ./target/release/chronos-rs infer \
      --gguf gguf/chronos-f16.gguf \
      --config models/config.json
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import chronos_rs

model = chronos_rs.Chronos("gguf/chronos-f16.gguf", "models/config.json")

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

Chronos-2 is an encoder-only bidirectional model:

- **Input**: Time series values are instance-normalized, patched, and concatenated with time encodings and observation masks
- **Encoder**: Alternating `TimeSelfAttention` + `GroupSelfAttention` + `FeedForward` blocks, all with T5-style RMSNorm
- **RoPE**: Standard Llama rotate_half (`[-x[half:], x[:half]]`), unlike Toto which uses xPos
- **Attention**: Scale = 1.0 (no `1/√d` scaling, per the original implementation)
- **Output**: Last `n_output_patches` hidden states → ResidualBlock → quantile predictions

For batch=1 (univariate inference), `GroupSelfAttention` reduces to a position-wise `v → o` projection.
