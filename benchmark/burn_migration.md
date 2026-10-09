# Burn migration measurements

Every number here comes from executed code, not estimates. Op microbenchmarks
run offline in CI (`cargo test -p zsfm-burn -- --nocapture`). Model-level
numbers come from `zsfm-bench` side-by-side runs once each model ports.

## Op parity (candle vs Burn Flex, CPU, Apple Silicon)

Source: `zsfm-rs/crates/zsfm-burn/tests/parity.rs` (50 iters, median).
Backend is Flex (pure-Rust CPU): `burn-ndarray` is deprecated in Burn 0.22,
so the spike targets Flex from the start.

| Op | Shape | candle ms | Burn ms | max abs err | Gate |
|----|-------|----------:|--------:|------------:|------|
| linear | 32x64→64 | 0.662 | 0.462 | 0.00e0 | <1e-5 ✅ |
| rms_norm | 32x64 | 0.208 | 0.139 | 1.19e-7 | <1e-4 ✅ |
| attention | b2 h4 sq16 hd32 | 1.345 | 0.914 | 2.98e-7 | <1e-4 ✅ |

Notes:

- Burn Flex beats candle on all three ops at these shapes (linear 1.4x, rms_norm 1.5x, attention 1.5x).
- Re-measure at model shapes (d=512..2048, longer sequences) before concluding; attention scaling with seq length matters most.
- Gate is max-abs-error on identical inputs. Model-level gate is stricter (see below).

## Model rollout tracker

Gate per model: same GGUF, same contexts, max-abs-error vs candle path
below 1e-5 on fixed probes AND no MAE regression in `zsfm-bench report`.

| Model | Status | Probe err | Bench MAE Δ | Notes |
|-------|--------|----------:|------------:|-------|
| ttm | planned pilot | — | — | small, univariate, no RoPE |
| timesfm | queued | — | — | — |
| sundial | queued | — | — | — |
| moment | queued | — | — | — |
| chronos | queued | — | — | RoPE family |
| toto | queued | — | — | RoPE + F64 paths |
| moirai | queued | — | — | — |
| moirai2 | queued | — | — | partial RoPE |
| lag_llama | queued | — | — | keep simdeez fast path |
| flowstate | queued | — | — | keep simdeez fast path |
| tirex | queued | — | — | keep simdeez fast path |
| mitra | queued | — | — | tabular attention |
| tabdpt | queued | — | — | — |
| tabicl | queued | — | — | — |
| tabpfn | queued | — | — | — |
| tabfm | queued | — | — | ensemble |

## Engine selection

`zsfm <model> infer --engine candle|burn` (default: `candle` until each model
flips after passing its gate). `--backend` selection arrives with Phase 5;
until then Burn runs Flex CPU. TTM already dispatches on the flag (Burn arm
errors clearly until the Phase 2 port lands).
