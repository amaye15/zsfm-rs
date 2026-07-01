---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model: ibm-granite/granite-timeseries-ttm-r2
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - mlp-mixer
  - rust
inference: false
---

# ttm-rs

Pure Rust converter and inference engine for [ibm-granite/granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).

Pre-converted GGUF files are available at [amaye15/ttm-gguf](https://huggingface.co/amaye15/ttm-gguf). Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads the model from HuggingFace and writes a GGUF file:

```bash
# F16 (recommended)
./target/release/ttm-rs convert --model ibm-granite/granite-timeseries-ttm-r2 --dtype f16 --output gguf/ttm-f16.gguf

# Q8_0 (smallest)
./target/release/ttm-rs convert --dtype q8 --output gguf/ttm-q8.gguf

# F32 (full precision)
./target/release/ttm-rs convert --dtype f32 --output gguf/ttm-f32.gguf
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

Print all tensor names and shapes from a safetensors file:

```bash
./target/release/ttm-rs inspect-tensors models/model.safetensors
```

## Infer

Run forecasting from comma-separated context values. The context must be at least `patch_length` steps (defined in `config.json`):

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 96}' \
  | ./target/release/ttm-rs infer \
      --gguf gguf/ttm-f16.gguf \
      --config models/config.json
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "ttm",
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
  | ./target/release/ttm-rs infer \
      --gguf gguf/ttm-f16.gguf \
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
import ttm_rs

model = ttm_rs.Ttm("gguf/ttm-f16.gguf", "models/config.json")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=96)
point  = result["choices"][0]["forecast"]["point"]

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=96)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

TTM (Tiny Time-Mixer) is a compact encoder-decoder model:

- **Encoder**: Multi-layer mixer blocks operating across the patch dimension; adaptive patching levels allow different temporal resolutions simultaneously
- **Decoder**: Lightweight projection from encoder representations to the forecast horizon
- **Scale**: ~1M parameters — orders of magnitude smaller than transformer-based foundation models, competitive on short-horizon benchmarks
- **Config**: `context_length`, `prediction_length`, `patch_length`, `patch_stride`, `d_model`, `num_layers`, `decoder_num_layers`, and `adaptive_patching_levels` are loaded from `config.json`
