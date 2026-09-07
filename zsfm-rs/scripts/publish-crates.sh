#!/usr/bin/env bash
set -euo pipefail

# Publishes every zsfm-rs crate to crates.io in dependency order (leaf crates
# first, `zsfm` last). Safe to re-run: a crate/version already live on
# crates.io is skipped rather than failing the whole run, so a partial
# failure can just be re-run once the underlying issue is fixed.
#
# Usage: CARGO_REGISTRY_TOKEN=<token> ./scripts/publish-crates.sh

cd "$(dirname "$0")/.."

CRATES=(
  zsfm-core
  zsfm-gguf
  zsfm-hub
  zsfm-nn
  zsfm-checkpoint
  zsfm-chronos
  zsfm-flowstate
  zsfm-lag-llama
  zsfm-mitra
  zsfm-moirai
  zsfm-moirai2
  zsfm-moment
  zsfm-sundial
  zsfm-tabdpt
  zsfm-tabfm
  zsfm-tabicl
  zsfm-tabpfn
  zsfm-timesfm
  zsfm-tirex
  zsfm-toto
  zsfm-ttm
  zsfm
)

VERSION=$(cargo metadata --no-deps --format-version 1 | python3 -c '
import json, sys
data = json.load(sys.stdin)
pkg = next(p for p in data["packages"] if p["name"] == "zsfm-core")
print(pkg["version"])
')

echo "Publishing zsfm-rs v$VERSION to crates.io ..."

# crates.io rate-limits publishing (a token-bucket burst, then a slow refill —
# see https://crates.io/docs/rate-limits). A long run publishing 22 crates back
# to back can burn through the burst allowance and get a 429; retry with
# exponential backoff rather than failing the whole release over it. Any other
# failure (compile error, bad token, etc.) fails immediately — no point
# retrying those.
publish_with_retry() {
  local crate="$1"
  local attempt=1
  local max_attempts=6
  local backoff=30
  local output

  while true; do
    if output=$(cargo publish -p "$crate" 2>&1); then
      echo "$output"
      return 0
    fi
    echo "$output"
    if echo "$output" | grep -qi "429 Too Many Requests\|too many new crates\|too many requests"; then
      if [ "$attempt" -ge "$max_attempts" ]; then
        echo "== $crate: still rate-limited after $max_attempts attempts, giving up =="
        return 1
      fi
      echo "== $crate: rate-limited by crates.io (attempt $attempt/$max_attempts), waiting ${backoff}s …"
      sleep "$backoff"
      backoff=$((backoff * 2))
      attempt=$((attempt + 1))
      continue
    fi
    echo "== $crate: publish failed for a non-rate-limit reason, giving up =="
    return 1
  done
}

for crate in "${CRATES[@]}"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" -A "zsfm-rs-publish-script (github.com/amaye15/zsfm-rs)" \
    "https://crates.io/api/v1/crates/$crate/$VERSION")
  if [ "$status" = "200" ]; then
    echo "== $crate $VERSION already published, skipping =="
    continue
  fi
  echo "== publishing $crate $VERSION =="
  if ! publish_with_retry "$crate"; then
    exit 1
  fi
  # crates.io's sparse index needs a moment to propagate before the next
  # crate's dependency resolution can see this one.
  sleep 20
done

echo "All crates published."
