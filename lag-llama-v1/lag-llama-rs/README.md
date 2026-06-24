---
license: mit
library_name: gguf
tags:
  - gguf
  - time-series
  - forecasting
  - rust
base_model: time-series-foundation-models/Lag-Llama
---

# lag-llama-rs

Pure Rust converter and inference engine for [time-series-foundation-models/Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama).

Produces GGUF v3 files and runs native forecasting — no Python required.

## Build

```bash
cargo build --release
```

## Convert

Downloads `lag-llama.ckpt` from HuggingFace and converts it directly via candle's pickle reader — no Python or intermediate extraction step required:

```bash
# F16 (recommended)
./target/release/lag-llama-rs convert --model time-series-foundation-models/Lag-Llama --dtype f16 --output gguf/lag_llama-f16.gguf

# Q8_0 (smallest)
./target/release/lag-llama-rs convert --dtype q8 --output gguf/lag_llama-q8.gguf

# F32 (full precision)
./target/release/lag-llama-rs convert --dtype f32 --output gguf/lag_llama-f32.gguf
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
./target/release/lag-llama-rs inspect-tensors models/model.safetensors
```

## Infer

Run point forecasting from comma-separated context values:

```bash
echo '{"context": [1.0, 1.2, 1.5, 1.3, 1.8, 2.0, 1.9, 2.1], "horizon": 96}' \
  | ./target/release/lag-llama-rs infer --gguf gguf/lag_llama-f16.gguf
```

Output is JSON in an OpenAI-compatible forecast format:

```json
{
  "id": "forecast-000001932b7a1234",
  "object": "forecast",
  "created": 1749686400,
  "model": "lag-llama",
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
  | ./target/release/lag-llama-rs infer --gguf gguf/lag_llama-f16.gguf
```


## Python bindings

Install with [maturin](https://github.com/PyO3/maturin) inside a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
```

```python
import lag_llama_rs

model = lag_llama_rs.LagLlama("gguf/lag_llama-f16.gguf")

result = model.forecast([1.0, 1.2, 1.5, 1.3, 1.8, 2.0], horizon=96)
point  = result["choices"][0]["forecast"]["point"]

# Batch — one Choice per series
result = model.forecast([[1.0, 1.2, 1.5], [2.0, 2.2, 2.5]], horizon=96)
```

`forecast` returns a Python dict in the same OpenAI-compatible format as the CLI.

## Architecture notes

Lag-Llama is a LLaMA-style autoregressive decoder for time series:

- **Input**: Lagged values of the target series are appended as covariates alongside the most-recent context, giving the model a multi-scale view of the history
- **Backbone**: Standard LLaMA decoder (RoPE, causal self-attention, SwiGLU FFN)
- **Output**: Point forecast for each future timestep via autoregressive decoding
- **Checkpoint**: Reads `.ckpt` (PyTorch Lightning pickle) directly — no Python required
