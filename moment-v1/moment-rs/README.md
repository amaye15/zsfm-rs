---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model: moment-research/MOMENT-1-large
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - transformer
  - masked-encoder
  - rust
inference: false
---

# moment-rs

Pure Rust converter and inference engine for [moment-research/MOMENT-1-large](https://huggingface.co/moment-research/MOMENT-1-large).

Pre-converted GGUF files are available at [amaye15/moment-gguf](https://huggingface.co/amaye15/moment-gguf). Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/moment-rs convert --model moment-research/MOMENT-1-large --dtype f16 --output gguf/moment-f16.gguf

# Q8_0 (smallest)
./target/release/moment-rs convert --dtype q8 --output gguf/moment-q8.gguf

# F32 (full precision)
./target/release/moment-rs convert --dtype f32 --output gguf/moment-f32.gguf
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
./target/release/moment-rs inspect-tensors models/model.safetensors
```

## Infer

Run forecasting from stdin JSON:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 96}' \
  | ./target/release/moment-rs infer --gguf gguf/moment-f16.gguf
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "moment",
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

**Batch / Multivariate inference** — Moment is channel-independent: each variate is encoded as an independent series. Pass a batch of univariate series to get one `Choice` per series, or use the batch mode to handle multiple variates of a multivariate dataset by submitting each variate as a separate item:

```bash
# Two independent series — one Choice each
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 96}' \
  | ./target/release/moment-rs infer --gguf gguf/moment-f16.gguf
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import moment_rs

model = moment_rs.Moment("gguf/moment-f16.gguf")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=96)
point  = result["choices"][0]["forecast"]["point"]

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=96)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

MOMENT-1-large is a masked patch encoder foundation model:

- **Input**: Time series is split into fixed-length patches; forecast-window patches are replaced with a learnable mask token during pretraining
- **Backbone**: T5-style bidirectional transformer with relative position biases; ~385M parameters
- **Pretraining**: Self-supervised masked patch reconstruction across diverse time series datasets
- **Output**: Patch-level representations decoded to point forecasts for each future timestep
