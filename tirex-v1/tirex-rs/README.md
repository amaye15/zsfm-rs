---
license: mit
library_name: gguf
pipeline_tag: time-series-forecasting
language:
  - en
base_model: NX-AI/TiRex
base_model_relation: quantized
quantized_by: amaye15
tags:
  - gguf
  - time-series
  - forecasting
  - zero-shot
  - probabilistic
  - slstm
  - rust
inference: false
---

# tirex-rs

Pure-Rust GGUF converter and inference engine for [TiRex](https://huggingface.co/NX-AI/TiRex) — a 35M parameter sLSTM-based zero-shot time-series forecasting model from NXAI.

Pre-converted GGUF files are available at [amaye15/tirex-gguf](https://huggingface.co/amaye15/tirex-gguf).

## Build

```bash
cargo build --release
```

## Convert

Download `NX-AI/TiRex` and convert to GGUF (reads `model.ckpt` directly — no Python required):

```bash
# All dtypes at once
bash scripts/convert_all.sh

# Single dtype
./target/release/tirex-rs convert --dtype f32 --output gguf/tirex-f32.gguf
./target/release/tirex-rs convert --dtype f16 --output gguf/tirex-f16.gguf
./target/release/tirex-rs convert --dtype q8  --output gguf/tirex-q8.gguf
```

Supported dtypes: `f32`, `f16`, `q8`.

## Infer

```bash
./target/release/tirex-rs infer \
  --gguf gguf/tirex-f32.gguf \
  --data "1.0,2.1,3.3,2.8,1.9,3.1,4.0,3.5" \
  --horizon 32
```

Add `--all-outputs` to include all 9 quantile forecasts (q0.1–q0.9) in the JSON response.

The output is OpenAI-compatible JSON:

```json
{
  "choices": [{
    "forecast": {
      "point": [...],
      "quantiles": { "0.10": [...], "0.50": [...], "0.90": [...] }
    }
  }]
}
```

## Architecture

TiRex is a 35M parameter sLSTM-based time-series foundation model:

- **Input**: Patch context of fixed length 2048 (left-padded with NaN if shorter), patch size 32 → 64 patches
- **Normalization**: Per-series StandardScaler (non-causal mean/std over full context)
- **Patch embedding**: `ResidualBlock(64→2048→512)` over concatenated [values | mask]
- **Backbone**: 12 × sLSTM blocks (sequential recurrence over 64 tokens)
  - Pre-RMSNorm → 4 headwise-linear gate projections (NH=4, DH=128) → sLSTM cell → MultiHeadLayerNorm → residual
  - Pre-RMSNorm → SiLU gated FFN (512→1408→512) → residual
- **Output**: `ResidualBlock(512→2048→288)` → 9 quantiles × 32 patch offsets per token
- **Decoding**: AR loop: take last token's prediction, extend context with NaN, repeat

## Python binding

```bash
uv add --dev maturin
uv run maturin develop --release
```

```python
import tirex_rs
model = tirex_rs.TiRex("gguf/tirex-f32.gguf")
result = model.forecast([1.0, 2.1, 3.3, 2.8, 1.9], horizon=32, all_outputs=True)
print(result["choices"][0]["forecast"]["point"])
```
