#!/usr/bin/env bash
# Upload all GGUF files to HuggingFace Hub.
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO="${HF_REPO:-amaye15/tirex-gguf}"
GGUF_DIR="./gguf"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN environment variable not set."
  exit 1
fi

for f in "$GGUF_DIR"/*.gguf; do
  huggingface-cli upload "$HF_REPO" "$f" "$(basename "$f")" \
    --token "$HF_TOKEN"
done
