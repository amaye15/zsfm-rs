#!/usr/bin/env bash
# Upload source + GGUF files to HuggingFace Hub via the built-in upload subcommand.
#
# Usage:
#   HF_TOKEN=hf_... ./scripts/upload.sh [--repo owner/repo-name]
#
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/moirai-2-rs

if [[ ! -f "$BIN" ]]; then
  echo "Binary not found — building release …"
  cargo build --release
fi

exec "$BIN" upload \
  --repo "amaye15/moirai-2-gguf" \
  ${HF_TOKEN:+--token "$HF_TOKEN"} \
  "$@"
