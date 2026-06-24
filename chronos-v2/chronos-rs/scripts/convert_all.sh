#!/usr/bin/env bash
# Download amazon/chronos-2 and convert to GGUF in all three dtypes.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/chronos-rs
MODEL=${1:-amazon/chronos-2}
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
  OUT="gguf/chronos-${DTYPE}.gguf"
  echo "=== Converting ${MODEL} → ${OUT} (dtype=${DTYPE}) ==="
  $BIN convert --model "$MODEL" --dtype "$DTYPE" \
    --model-dir "$MODEL_DIR" --output "$OUT" $TOKEN_ARG
  ls -lh "$OUT"
done

echo "Done. Files:"
ls -lh gguf/chronos-*.gguf
