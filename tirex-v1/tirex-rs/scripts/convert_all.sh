#!/usr/bin/env bash
# Download NX-AI/TiRex and convert to GGUF in all dtypes.
# Reads model.ckpt directly via candle's pickle reader — no Python required.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/tirex-rs
MODEL=${1:-NX-AI/TiRex}
MODEL_DIR=models
TOKEN_ARG=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARG="--token $HF_TOKEN"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found — run: cargo build --release"
  exit 1
fi

mkdir -p gguf

for DTYPE in f32 f16 q8; do
  OUT="gguf/tirex-${DTYPE}.gguf"
  echo "=== Converting ${MODEL} → ${OUT} (dtype=${DTYPE}) ==="
  $BIN convert --model "$MODEL" --dtype "$DTYPE" \
    --model-dir "$MODEL_DIR" --output "$OUT" $TOKEN_ARG
  ls -lh "$OUT"
done

echo "Done. Files:"
ls -lh gguf/tirex-*.gguf
