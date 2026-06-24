---
license: mit
library_name: gguf
tags:
  - gguf
  - time-series
  - forecasting
  - rust
base_model: thuml/sundial-base-128m
---

# sundial-rs

Pure Rust converter and inference engine for [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m).

Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/sundial-rs convert --model thuml/sundial-base-128m --dtype f16 --output gguf/sundial-f16.gguf

# Q8_0 (smallest)
./target/release/sundial-rs convert --dtype q8 --output gguf/sundial-q8.gguf

# F32 (full precision)
./target/release/sundial-rs convert --dtype f32 --output gguf/sundial-f32.gguf
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
./target/release/sundial-rs inspect-tensors gguf/sundial-f16.gguf
```

## Infer

Run forecasting from comma-separated context values:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 96}' \
  | ./target/release/sundial-rs infer --gguf gguf/sundial-f16.gguf
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "sundial",
  "choices": [{
    "index": 0,
    "forecast": {
      "point": [2.1, 2.3, 2.5, "..."],
      "quantiles": {}
    },
    "finish_reason": "stop"
  }],
  "usage": {"context_length": 8, "forecast_length": 96}
}
```

**Batch inference** — pass multiple series as a nested array to get one `Choice` per series:

```bash
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 96}' \
  | ./target/release/sundial-rs infer --gguf gguf/sundial-f16.gguf
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import sundial_rs

model = sundial_rs.Sundial("gguf/sundial-f16.gguf")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=96)
point  = result["choices"][0]["forecast"]["point"]

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=96)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

Sundial is a flow-matching generative model for time series:

- **Encoder**: Patch-based causal transformer (128M params) that encodes the context window into a conditioning vector
- **Decoder**: A small flow network trained to map Gaussian noise to the forecast distribution, conditioned on the encoder output
- **Sampler**: Euler integration from t=0 (noise) to t=1 (data); Heun's method used for Python reference validation
- **RoPE**: Standard rotary positional embeddings applied to all attention heads
