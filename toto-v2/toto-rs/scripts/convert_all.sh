#!/usr/bin/env bash
# Convert all Toto-2.0 model sizes to F32, F16, and Q8_0 GGUF.
#
# Usage:
#   ./scripts/convert_all.sh              # convert everything
#   ./scripts/convert_all.sh toto-4m      # only convert the 4m model
#   HF_TOKEN=hf_xxx ./scripts/convert_all.sh
#
# Outputs go to: gguf/<tag>-<dtype>.gguf
# Model weights are cached in: models/<tag>/
set -euo pipefail

BIN="./target/release/toto-rs"
GGUF_DIR="./gguf"
MODELS_DIR="./models"

# model_id:tag pairs
MODELS=(
    "Datadog/Toto-2.0-4m:toto-4m"
    "Datadog/Toto-2.0-22m:toto-22m"
    "Datadog/Toto-2.0-313m:toto-313m"
    "Datadog/Toto-2.0-1B:toto-1b"
    "Datadog/Toto-2.0-2.5B:toto-2.5b"
)
DTYPES=(f32 f16 q8)

FILTER="${1:-}"  # optional: restrict to one tag

mkdir -p "$GGUF_DIR"

for entry in "${MODELS[@]}"; do
    model_id="${entry%%:*}"
    tag="${entry##*:}"

    if [[ -n "$FILTER" && "$tag" != "$FILTER" ]]; then
        continue
    fi

    model_dir="${MODELS_DIR}/${tag}"
    mkdir -p "$model_dir"

    for dtype in "${DTYPES[@]}"; do
        output="${GGUF_DIR}/${tag}-${dtype}.gguf"

        if [[ -f "$output" ]]; then
            echo "[skip] $output already exists"
            continue
        fi

        echo ""
        echo "=== $model_id  →  $dtype  →  $output ==="
        "$BIN" convert \
            --model "$model_id" \
            --model-dir "$model_dir" \
            --output "$output" \
            --dtype "$dtype" \
            ${HF_TOKEN:+--token "$HF_TOKEN"}
    done
done

echo ""
echo "All done. GGUF files:"
ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null || echo "  (none found)"
