#!/usr/bin/env bash
# Download thuml/sundial-base-128m and convert to GGUF in all dtypes.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/sundial-rs
MODEL=${1:-thuml/sundial-base-128m}
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
  OUT="gguf/sundial-${DTYPE}.gguf"
  echo "=== Converting ${MODEL} → ${OUT} (dtype=${DTYPE}) ==="
  $BIN convert --model "$MODEL" --dtype "$DTYPE" \
    --output "$OUT" $TOKEN_ARG
  ls -lh "$OUT"
done

echo "Done. Files:"
ls -lh gguf/sundial-*.gguf
