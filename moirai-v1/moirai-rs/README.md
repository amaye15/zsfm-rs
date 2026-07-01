---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model: Salesforce/moirai-1.0-R-large
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - transformer
  - universal-forecasting
  - rust
inference: false
---

# moirai-rs

Pure Rust converter and inference engine for [Salesforce/moirai-1.0-R-large](https://huggingface.co/Salesforce/moirai-1.0-R-large).

Pre-converted GGUF files are available at [amaye15/moirai-gguf](https://huggingface.co/amaye15/moirai-gguf). Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/moirai-rs convert --model Salesforce/moirai-1.0-R-large --dtype f16 --output gguf/moirai-f16.gguf

# Q8_0 (smallest)
./target/release/moirai-rs convert --dtype q8 --output gguf/moirai-q8.gguf

# F32 (full precision)
./target/release/moirai-rs convert --dtype f32 --output gguf/moirai-f32.gguf
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
./target/release/moirai-rs inspect-tensors models/model.safetensors
```

## Infer

Run forecasting from stdin JSON:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 96}' \
  | ./target/release/moirai-rs infer --gguf gguf/moirai-f16.gguf
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "moirai",
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
  | ./target/release/moirai-rs infer --gguf gguf/moirai-f16.gguf
```

**Multivariate inference** — pass a 3D context array `[batch][variate][time]` to get a `variates` array in each choice. Each variate is processed independently (channel-independent):

```bash
echo '{
  "context": [
    [[1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1],
     [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
  ],
  "horizon": 96
}' \
  | ./target/release/moirai-rs infer --gguf gguf/moirai-f16.gguf
```

```json
{
  "choices": [{
    "index": 0,
    "forecast": {
      "variates": [
        {"point": [2.1, 2.3, "..."], "quantiles": {}},
        {"point": [1.3, 1.4, "..."], "quantiles": {}}
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
import moirai_rs

model = moirai_rs.Moirai("gguf/moirai-f16.gguf")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=96)
point  = result["choices"][0]["forecast"]["point"]

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=96)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

Moirai-1.0-R-large is a universal forecasting foundation model:

- **Input**: Multi-patch-size projections — patches at sizes 8, 16, 32, 64, and 128 are embedded in parallel, giving the encoder a multi-scale view of the context
- **Backbone**: Bidirectional encoder (attends over the full context window); ~311M parameters
- **Normalization**: Mean-absolute-mean (absmean) instance normalization
- **Output**: Student-t distribution head producing probabilistic point forecasts
