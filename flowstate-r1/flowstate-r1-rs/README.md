---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model: ibm-granite/granite-timeseries-flowstate-r1
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - probabilistic
  - state-space-model
  - rust
inference: false
---

# flowstate-r1-rs

Pure Rust converter and inference engine for [ibm-granite/granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1).

Pre-converted GGUF files are available at [amaye15/flowstate-r1-gguf](https://huggingface.co/amaye15/flowstate-r1-gguf). Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/flowstate-r1-rs convert --model ibm-granite/granite-timeseries-flowstate-r1 --dtype f16 --output gguf/flowstate-r1-f16.gguf

# Q8_0 (smallest)
./target/release/flowstate-r1-rs convert --dtype q8 --output gguf/flowstate-r1-q8.gguf

# F32 (full precision)
./target/release/flowstate-r1-rs convert --dtype f32 --output gguf/flowstate-r1-f32.gguf
```

To convert all dtypes at once:

```bash
./scripts/convert_all.sh
```

HuggingFace token (for gated models):

```bash
HF_TOKEN=hf_... ./scripts/convert_all.sh
```

## Inspect tensors

Print all tensor names and shapes from a `.safetensors` checkpoint or GGUF file:

```bash
./target/release/flowstate-r1-rs inspect-tensors models/model.safetensors
./target/release/flowstate-r1-rs inspect-tensors gguf/flowstate-r1-f16.gguf
```

## Dump basis

Dump the Legendre polynomial basis matrix to CSV for validation against the Python reference:

```bash
./target/release/flowstate-r1-rs dump-basis --config models/config.json --prediction-length 24
```

## Infer

Run univariate quantile forecasting from a GGUF file:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 24}' \
  | ./target/release/flowstate-r1-rs infer \
      --gguf gguf/flowstate-r1-f16.gguf \
      --config models/config.json
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "flowstate",
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
  "usage": {"context_length": 8, "forecast_length": 24}
}
```

`point` is the median (q0.5) forecast; all quantile levels from `config.json` are included.

**Batch inference** — pass multiple series as a nested array to get one `Choice` per series:

```bash
echo '{"context": [[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], "horizon": 24}' \
  | ./target/release/flowstate-r1-rs infer \
      --gguf gguf/flowstate-r1-f16.gguf \
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
import flowstate_r1_rs

model = flowstate_r1_rs.FlowState("gguf/flowstate-r1-f16.gguf", "models/config.json")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=24)
fc     = result["choices"][0]["forecast"]
point  = fc["point"]           # median forecast
q10    = fc["quantiles"]["0.10"]  # 10th-percentile
q90    = fc["quantiles"]["0.90"]  # 90th-percentile

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=24)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

FlowState-R1 is an encoder-decoder model:

- **Encoder**: State-space model backbone that encodes the context window into a conditioning latent
- **Decoder**: Projects the latent through a Legendre polynomial basis to produce quantile forecasts at arbitrary horizons
- **Output**: Multi-quantile forecasts (one row per future timestep, one column per quantile level)
