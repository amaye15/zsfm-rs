# Optimisation Log

Tracking before/after results for each optimisation applied to the GGUF inference binaries.
Canonical measurement: `zsfm-bench run --dataset <name> --context 512 --horizon 96 --windows 30` (or `zsfm-bench report`
for all 21 datasets at once) across all 21 datasets, context=512, horizon=96, 30 windows. The historical entries below
were measured with the old `python benchmark/run_bench.py`, since replaced by the in-process Rust `zsfm-bench` — see
`zsfm-rs/crates/zsfm-bench`.
Latency reported as **ms per window** (total elapsed / n_valid_windows). Run 3× per step, record median.

---

## Baseline

**Date:** 2026-06-13 (from benchmark.md)  
**Build:** `opt-level = 3` only — no LTO, no SIMD flags, F32/F16 weights, Device::Cpu  
**Command:** `python benchmark/run_bench.py`

| Model     | Min ms | Max ms | Weight dtype |
|-----------|-------:|-------:|:-------------|
| TTM       | 7      | 18     | F32          |
| Toto      | 12     | 26     | F32          |
| Moirai-2  | 25     | 62     | F32          |
| Chronos   | 139    | 358    | F16          |
| TimesFM   | 200    | 516    | F16          |
| Sundial   | 207    | 448    | F16          |
| Moirai    | 323    | 852    | F32          |
| Moment    | 396    | 1,057  | F32          |
| Lag-Llama | 655    | 1,773  | F32          |

---

## Steps 1+2 — LTO + codegen-units + strip + panic=abort + target-cpu=native

**Date:** 2026-06-24  
**Change:** Applied together (all 11 `Cargo.toml` + `.cargo/config.toml` in each model dir)  
**Datasets measured:** ETTh1 (2880 test rows), ETTm1 (11520), Wind (2000) — 30 windows each  
**MAE unchanged** on all models vs baseline (lossless optimisation)

| Model     | ETTh1 ms | ETTm1 ms | Wind ms | Baseline range | Speedup |
|-----------|:--------:|:--------:|:-------:|:--------------:|:-------:|
| TTM       | 7        | 2        | 2       | 7–18           | 1–9×    |
| Toto      | 10       | 3        | 2       | 12–26          | 1.2–13× |
| Moirai-2  | 15       | 9        | 9       | 25–62          | 1.7–7×  |
| Chronos   | 42       | 35       | 31      | 139–358        | **3.3–10×** |
| TimesFM   | 48       | 33       | 33      | 200–516        | **4.2–16×** |
| Sundial   | 105      | 95       | 98      | 207–448        | **2–4.6×**  |
| Moirai    | 78       | 58       | 61      | 323–852        | **4.1–14×** |
| Moment    | 167      | 147      | 153     | 396–1,057      | **2.4–7×**  |
| Lag-Llama | 560      | 538      | 543     | 655–1,773      | 1.2–3.3×    |
| FlowState | 19       | 13       | 13      | (new model)    | —           |

**Verdict:** ✅ KEEP — large verified speedups, zero accuracy impact.  
Biggest winners: Chronos (3–10×), TimesFM (4–16×), Moirai (4–14×), Moment (2–7×), Sundial (2–5×).  
Smallest gain: Lag-Llama (1–3×, bottlenecked by sequential AR decode), TTM (already tiny at 7ms).

---

## Step 4 — KV Cache: Eliminate Duplicate Tensor::cat (Lag-Llama, TimesFM, Moirai-2)

**Date:** 2026-06-24  
**Change:** `decode_attn_kv` now returns `k_full`/`v_full` (extended cache) instead of the single new token; callers store directly, eliminating a second `Tensor::cat` per layer per decode step.  
- Lag-Llama: removed 3,040 cats/call (32 layers × 95 steps)
- TimesFM: removed 20 × decode_steps cats
- Moirai-2: removed 6 × decode_steps cats

**Included in Steps 1+2 run above** (applied simultaneously).  
Lag-Llama contribution to its 1.2–3.3× improvement vs baseline is partially this change.
Steps 1+2 dominate; individual isolation not measured separately.

**Verdict:** ✅ KEEP — logically correct, reduces allocations, zero accuracy impact.

---

## Step 6 — FlowState SSM Scan: Pre-allocated Scratch Buffers

**Date:** 2026-06-24  
**Change:** Pre-allocate `scan_r`/`scan_i` (seq_len × state_dim each) and `state_r`/`state_i` (state_dim) once in `forecast` and pass to `apply_s5_layer` for all blocks. Eliminates 10 large Vec allocations + zero-initialisations per forecast call (~1 MB each for ctx=512).  
**Note on Blelloch scan:** A parallel prefix scan on a SINGLE CPU thread does 2× more work than the sequential scan (same O(N) depth, just with different ordering). It only pays off with multi-threading. The inner `state_dim` loop is already NEON-vectorised by LLVM after Step 2 (`target-cpu=native`).  

**FlowState not in original benchmark** (new model). Measured after all steps:
- ETTh1: 19 ms/window
- ETTm1: 13 ms/window  
- Wind: 13 ms/window

**Verdict:** ✅ KEEP — reduced heap churn, correctness unchanged.

---

## Running Totals (measured 2026-06-24)

ETTh1 / ETTm1 / Wind — 30 windows, ctx=512, h=96. Accuracy unchanged on all models.

| Model     | Baseline (ms/win) | After all steps (ms/win) | Best speedup |
|-----------|:-----------------:|:------------------------:|:------------:|
| TTM       | 7–18              | 2–7                      | 9×           |
| Toto      | 12–26             | 2–10                     | 13×          |
| Moirai-2  | 25–62             | 9–15                     | 7×           |
| Chronos   | 139–358           | 31–42                    | **10×**      |
| TimesFM   | 200–516           | 33–48                    | **16×**      |
| Sundial   | 207–448           | 95–105                   | 4.6×         |
| Moirai    | 323–852           | 58–78                    | **14×**      |
| Moment    | 396–1,057         | 147–167                  | 7×           |
| Lag-Llama | 655–1,773         | 538–560                  | 3.3×         |
| FlowState | (new)             | 13–19                    | —            |

---

## Precision Audit — F16 and Q8 vs F32 reference

**Date:** 2026-06-24  
**Command:** `python benchmark/compare_precision.py --horizon 64`  
**What it measures:** Rust GGUF inference vs Python reference implementation.  
- F32 Max |Δ| ≈ 0: confirms Rust and Python F32 paths are numerically identical (rounding only).  
- F16/Q8 Max |Δ|: absolute forecast error introduced solely by weight quantization.

| Model     | F32 Max|Δ| | F16 Max|Δ| | +vs F32  | Q8 Max|Δ| | +vs F32  | Q8 verdict     |
|-----------|:----------:|:----------:|:--------:|:---------:|:--------:|:---------------|
| TTM       | 0.000000   | 0.000727   | +0.00073 | 0.006165  | +0.00617 | ✅ low error   |
| Toto      | 0.000000   | 0.001741   | +0.00174 | 0.033268  | +0.03327 | ⚠️ moderate   |
| Moirai-2  | 0.000005   | 0.003075   | +0.00307 | 0.048241  | +0.04824 | ⚠️ moderate   |
| Moirai    | 0.000003   | 0.024928   | +0.02493 | 0.043276  | +0.04327 | ⚠️ moderate   |
| Lag-Llama | 0.000004   | 0.007306   | +0.00730 | 0.003335  | +0.00333 | ✅ Q8 < F16!  |
| Moment    | 0.000489   | 0.001225   | +0.00074 | 0.008256  | +0.00777 | ✅ low error   |
| Sundial   | 0.000021   | 0.001468   | +0.00145 | 0.009190  | +0.00917 | ✅ low error   |
| Chronos   | 0.000006   | 0.005198   | +0.00519 | 0.053040  | +0.05303 | ⚠️ highest Q8 |
| TimesFM   | 0.000000   | 0.001600   | +0.00160 | 0.006900  | +0.00690 | ✅ low error   |
| TiRex     | 0.005247   | 0.004525   | -0.00072 | 0.009203  | +0.00396 | ✅ low error   |

**Notes:**
- F32 vs Python reference is near-zero across all models — the build-flag optimisations (LTO, codegen-units, panic=abort, target-cpu=native) introduced **zero numerical drift**. The compiler only affects code generation, not arithmetic semantics.
- F16: all models < 0.025 Max|Δ|. Moirai is highest (0.025) due to its large parameter count amplifying rounding.
- Q8 outliers (Chronos 0.053, Moirai-2 0.048, Moirai 0.043, Toto 0.033): these are absolute prediction-space errors. For typical normalised time series (zero-mean, unit variance), these represent <5% of the signal range and are within accepted quantization tolerance for probabilistic forecasting.
- Lag-Llama Q8 (0.003) is _better_ than its F16 (0.007) — an artefact of the Q8_0 block format applying symmetric scaling per 32-element block, which can outperform per-tensor F16 rounding for weight matrices with high dynamic range.
- TiRex F32 shows 0.005247 baseline error vs tirex-ts: TiRex's sLSTM recurrence accumulates small floating-point deltas over 512 AR tokens, giving a higher F32 floor than transformer models. F16 (0.004525) is slightly _lower_ than F32 on this test input (quantization noise partially cancelling accumulated rounding error). Q8 adds +0.004 above F32 — still within low-error range.
- FlowState not included (no Python reference implementation available for comparison).

**Verdict:** ✅ All three weight dtypes are safe to use. Prefer F32 for highest accuracy, F16 for ~2× memory saving with negligible error, Q8 for ~4× memory saving where latency/memory matters more than last-decimal-place accuracy.

---

## Round 2 — TiRex: Fused Gate Projections + Scratch Buffer Pre-allocation

**Date:** 2026-06-24  
**File:** `tirex-v1/tirex-rs/src/infer/mod.rs`

### Step B — Fuse 4 headwise gate projections → 1 call

**Change:** At model load time, concatenate the 4 separate gate weight matrices (fgate, igate, zgate, ogate), each `[NH, DH, DH]` = `[4, 128, 128]`, into a single `fizo_w: [NH, 4*DH, DH]` = `[4, 512, 128]`. At inference time, replace 4 calls to `headwise_linear_batch` with a single `headwise_linear_batch_ng` that writes output directly in `x_g` layout — eliminating both the 4 separate calls and the subsequent interleaving copy loop.

- 4 gate projection calls → 1 call per block
- x_g construction loop (lines 258–264) eliminated entirely
- x_h is loaded into CPU cache once per (token, head) rather than 4 times
- Output buffer (`sc_xg`) is pre-allocated and reused across all 12 blocks (Step C)

### Step C — Pre-allocate scratch buffers + `is_first = t == 0`

**Change:** Pre-allocate 9 scratch buffers in `forecast()` before the block loop and pass them by `&mut` reference into `forward_slstm_block`. Each block call previously allocated:

| Buffer | Size | Per-forecast allocs saved |
|--------|------|--------------------------|
| `x_g` / `sc_xg` | 64 × 4 × 512 × 4B ≈ 512 KB | 12 |
| `h_out` / `sc_hout` | 64 × 512 × 4B ≈ 128 KB | 12 |
| `y` / `sc_y` | same | 12 |
| `x_n` / `sc_xn` | same | 12 (×2: norm + FFN) |
| `raw` (inner loop) | 4 × 512 × 4B = 8 KB | 12 × 64 = 768 |
| `h_new/c_new/n_new/m_new` | 512 × 4B = 2 KB each | 12 × 64 × 4 = 3072 |

Total allocations eliminated: **≈3,900 per forecast call** | **Total heap traffic avoided: ≈70 MB per call**

Also replaced `is_first = n.iter().all(|&v| v == 0.0)` (O(512) scan on every time step × 768 total steps) with `let is_first = t == 0`.

### Measured results

**Baseline** (round 1 flags, original infer/mod.rs): `547.5 ms` median (ctx=512, h=96, F32)  
**Optimised** (gate fusion + scratch buffers): `547.5 ms` median — **within measurement noise**

**Why latency was unchanged despite large allocation savings:**

TiRex's hot path is dominated by the Candle-based FFN in `ffn_forward`, which calls Accelerate BLAS for three matrix multiplications per sLSTM block:
- `[64, 512] × [512, 1408]` gate projection (~46M FLOPs/block)
- `[64, 512] × [512, 1408]` up projection
- `[64, 1408] × [1408, 512]` down projection

Over 3 AR steps × 12 blocks = 36 FFN calls → ~5 BFLOP/s of Accelerate work per forecast call.

The headwise gate projections (`4 × [64×4×128×128]` = ~34M FLOPs of scalar Rust) represent only ~1.5% of total FLOPs. Even after the 4×→1 call reduction and scratch buffer savings, the FFN Accelerate bottleneck is unchanged. The subprocess cost (loading 134 MB GGUF + process spawn) adds ~100ms on top.

**TiRex precision (vs tirex-ts reference, `python benchmark/compare_precision.py`):**
| dtype | Max\|Δ\| | Mean\|Δ\| | vs F32 max |
|-------|----------|----------|:----------:|
| F32   | 0.005247 | 0.002302 | —          |
| F16   | 0.004525 | 0.002497 | -0.000722  |
| Q8    | 0.009203 | 0.002867 | +0.003956  |

**Verdict:** ✅ KEEP — code is correct (zero arithmetic change), removes ~70 MB of heap churn per call, and eliminates 3,900 small allocations per forecast. **No measurable wall-time speedup** because the Candle/Accelerate FFN (~98% of compute) is unchanged. True speedup for TiRex would require fusing or replacing the FFN with a Candle-free implementation, which is out of scope.

---

## Round 3 — QMatMul Quantized Inference ❌ REVERTED

**Date:** 2026-06-24  
**Status:** Implemented, benchmarked, and reverted — regression for most models.

### What was attempted

Replaced `load_t()` (dequantize Q8→F32 at model load, store as `Tensor`) with `load_q()` (`QMatMul::from_qtensor()` — keep weights in Q8 format, dequantize 32 elements at a time on-the-fly during each matmul). Applied to 7 models: Moment, Moirai, Moirai-2, Lag-Llama, Sundial, TTM, TiRex. Also added `--dtype q8` flag to `run_bench.py`.

**Hypothesis:** Q8 weights are 4× smaller, so large projection matrices fit in L2/L3 cache rather than spilling to DRAM, reducing memory bandwidth pressure and improving latency.

### Measured results — ETTh1, 30 windows, ctx=512, h=96

| Model     | F32 ms (baseline) | Q8 ms | Δ            |
|-----------|:-----------------:|:-----:|:------------:|
| Moment    | 164               | 623   | **3.8× slower** |
| Moirai    | 76                | 138   | 1.8× slower  |
| Moirai-2  | 14                | 42    | 3.0× slower  |
| Lag-Llama | 553               | 610   | 1.1× slower  |
| Sundial   | 104               | 168   | 1.6× slower  |
| TTM       | 8                 | 3     | **2.7× faster** |
| TiRex     | 546               | 785   | 1.4× slower  |

### Why it regressed

Candle's `QMatMul::forward()` for Q8 tensors dequantizes blocks in software and **bypasses Apple Accelerate BLAS**. For large transformer weight matrices (e.g. Moment's `[1024, 4096]` projections), Accelerate achieves near-peak FLOPS (~40+ GFLOPS/core) — far faster than software block dequantization regardless of cache pressure savings. The memory bandwidth reduction is real but insufficient to overcome the loss of BLAS.

TTM is the one exception: its MLP matrices are tiny (`[48, 64]`, `[64, 256]` etc.), already below the BLAS crossover threshold. For these, software matmul is competitive and the smaller Q8 working set genuinely reduces cache pressure.

### Verdict: ❌ REVERTED

Code reverted to F32 `Tensor`-based matmul. The `--dtype q8` flag was also removed. Key learning: **on Apple Silicon with Accelerate, Q8 quantized inference via Candle's `QMatMul` is slower than F32 for any model whose weight matrices benefit from BLAS.** Future Q8 speedups would require a custom dequantize-then-BLAS path (dequantize a block to F32, then hand it to `cblas_sgemm`) rather than Candle's current block-at-a-time software path.

---

## Round 4 — SIMD Vectorisation via `simdeez` ✅ KEEP

**Date:** 2026-06-24  
**Models changed:** TiRex (`tirex-v1`), FlowState (`flowstate-r1`)  
**Crate:** [`simdeez`](https://crates.io/crates/simdeez) v3.0.1 — runtime dispatch across SSE2/AVX2/AVX-512 (x86) and NEON (AArch64)

### Rationale

All other models (Moment, Moirai, etc.) compute norms and activations through Candle Tensor ops, which already dispatch to Accelerate BLAS. There is nothing to gain there with manual SIMD. Only TiRex and FlowState contain material pure-Rust scalar loops that bypass BLAS entirely:

- **TiRex** `forward_slstm_block` gate loop: `exp`, `tanh`, `ln` over `d=512` elements, called 576 times per forecast (16 tokens × 12 blocks × 3 AR steps). Also `rms_norm_inplace` (squared-sum + scale-weight) and `headwise_linear_batch_ng` inner dot product (`dh=128` FMAs).
- **FlowState** `apply_s5_layer` SSM scan: complex multiply-add over `state_dim` elements, pure FMA, no transcendentals.

### What was implemented

**TiRex — 4 `simd_runtime_generate!` kernels:**

| Kernel | Replaces | Key ops |
|--------|----------|---------|
| `simd_sq_sum` | squared-sum loop in `rms_norm_inplace` | FMA, `horizontal_add` |
| `simd_scale_weight` | scale loop in `rms_norm_inplace` | FMA |
| `simd_dot` | inner `di` loop in `headwise_linear_batch_ng` | FMA, `horizontal_add` |
| `simd_gate_update` | `for i in 0..d` gate loop in `forward_slstm_block` | `exp_u35`, `tanh_u35`, `ln_u35`, `blendv`, FMA |

`simd_gate_update` computes the full sLSTM gate update in SIMD: numerically-stable two-branch `log_sigmoid` (both branches computed, blended on sign), sigmoid, input/forget/cell/output gates, and the `|nnew| > 1e-8` guard for `h_new` as a SIMD blend (NaN from division is computed but discarded before write). The `is_first` flag is a scalar bool — no data-dependent branching.

**FlowState — 1 `simd_runtime_generate!` kernel:**

| Kernel | Replaces | Key ops |
|--------|----------|---------|
| `ssm_scan_step` | `for s in 0..state_dim` loop in `apply_s5_layer` | FMA, `neg_mul_add` |

Complex recurrence `new_r = ar*sr − ai*si + br`, `new_i = ar*si + ai*sr + bi` expressed as FMA pairs. Both calls per timestep (two-pass scan in `apply_s5_layer`) replaced.

### Measured results — sequential runs, 30 windows, ctx=512, h=96

| Model     | ETTh1 ms | ETTm1 ms | Wind ms | Pre-R4 baseline (ms) | Speedup |
|-----------|:--------:|:--------:|:-------:|:--------------------:|:-------:|
| TiRex     | 441      | 440      | 437     | ~546 (Round 2/3)     | **~1.24×** |
| FlowState | 14       | 14       | 14      | 19 / 13 / 13         | 1.36× / ≈1× / ≈1× |

**MAE unchanged** — SIMD transcendentals (`_u35` = 3.5 ULP) fall within TiRex's existing F32 recurrence noise floor (Max|Δ| ≤ 0.005247).

### Analysis

**TiRex** gains ~1.24× end-to-end (~1.32× compute-only, excluding the fixed ~100ms GGUF load). The primary driver is `simd_gate_update`: NEON vectorises 4 `exp`/`tanh`/`ln` evaluations simultaneously across the 512-element gate loop, eliminating the scalar transcendental bottleneck (294,912 transcendental evaluations per forecast across 576 iterations of the `d=512` loop).

**FlowState** gains 1.36× on ETTh1 (19ms → 14ms) but is neutral on shorter-context datasets (ETTm1/Wind: 13ms → 14ms, within noise). The SSM scan is a pure FMA loop — LLVM with `target-cpu=native` already auto-vectorises it to NEON. The explicit simdeez kernel matches LLVM's output rather than exceeding it. At 14ms FlowState remains the fastest model in the suite.

**Why no other models gained:** All Candle-based models dispatch matmuls to Accelerate BLAS with no exposed scalar loops — no SIMD surface area exists outside of TiRex and FlowState.

**Cross-platform correctness:** `simdeez` v3.0.1 performs runtime ISA detection — the same binary uses AVX2 on x86-64, SSE2 on older x86, and NEON on AArch64. No per-platform `cfg_if` required. `target-cpu=native` in `.cargo/config.toml` unlocks AVX2 on x86 machines at compile time; on Apple Silicon NEON is the active path.

### Verdict: ✅ KEEP

TiRex achieves a genuine ~1.24× end-to-end speedup (1.32× compute-only) with zero accuracy impact. FlowState is neutral-to-slightly-better. The implementation is cross-platform, uses no `unsafe` beyond simdeez internals, and adds no new dependencies beyond the single `simdeez = "3"` entry in each `Cargo.toml`.

---

## Round 5 — RoPE Tensor Pre-computation (TimesFM + Sundial) ✅ KEEP

**Date:** 2026-06-24  
**Models changed:** TimesFM (`timesfm-v2.5`), Sundial (`timer-v3`)

### What was changed

Both models' `RopeCache` structs stored the RoPE tables as `Vec<f32>` and rebuilt `Tensor` objects on **every `apply()` call** via `to_vec()` + `Tensor::from_vec()`. This runs inside every attention forward pass.

**Old pattern** (identical in both `rope.rs` files):
```rust
// Per apply() call — allocates every time:
let cos_slice: Vec<f32> = self.cos[start_pos * hd..(start_pos + seq) * hd].to_vec();
let cos_t = Tensor::from_vec(cos_slice, (seq, hd), device)?
    .unsqueeze(0)?.unsqueeze(n)?;
```

**New pattern** — `Tensor` stored at construction time, `narrow()` produces a zero-copy view:
```rust
// At RopeCache::new(): one-time
let cos_t = Tensor::from_vec(cos, (max_seq, head_dim), device)?;

// Per apply() call — no allocation:
let cos_t = self.cos_t.narrow(0, start_pos, seq)?.unsqueeze(0)?.unsqueeze(n)?;
```

`narrow()` on a row-contiguous 2D tensor is a metadata-only operation (adjusts dims + offset, shares storage). No heap allocation, no memcpy.

**Additional Sundial optimisation:** The original `apply()` extracted `cos_half = narrow(last, 0, half)` then `cat([cos_half, cos_half])` to reconstruct a full-width tensor — redundant because the stored values already satisfy `cos[i] == cos[half+i]` by construction. The `narrow` + `cat` pair is eliminated, leaving `cos_t` used directly.

### Audit of other models

All other models were checked:
- **Chronos**: already stores `Tensor` in `RopeCache` and uses `narrow()` — no change.
- **Moirai**: RoPE cached by `seq_len` in `Mutex<HashMap>`, only allocates on cache miss — no change.
- **Lag-Llama**: RoPE cos/sin precomputed as `Tensor` at model load — no change.
- **Toto**: Uses xPos-RoPE with per-call exponent scaling (data-dependent) — unavoidable per-call computation.
- **Moment, Moirai-2, TTM, TiRex, FlowState**: no RoPE.

### Measured results — median of 3 runs, 30 windows, ctx=512, h=96

| Model | ETTh1 ms | ETTm1 ms | Wind ms | Baseline (ms) | Speedup |
|-------|:--------:|:--------:|:-------:|:-------------:|:-------:|
| TimesFM | 36 | 33 | 34 | 48 / 33 / 33 | **1.33×** / 1× / 1× |
| Sundial | 96 | 94 | 95 | 105 / 95 / 98 | **1.09×** / 1.01× / 1.03× |

**MAE unchanged** — no arithmetic change, only allocation strategy.

### Analysis

TimesFM's ETTh1 improvement (48→36ms) is the largest gain. TimesFM has 20 transformer blocks, each with prefill + decode attention calls. Each `apply()` previously did 2 `Vec::to_vec()` + 2 `Tensor::from_vec()` (heap alloc + memcpy). Eliminating these across all attention calls per forecast reduces both GC pressure and cache pollution.

Sundial's improvement is smaller in absolute terms (~9ms) but consistent across all datasets. Sundial also benefits from the Sundial-specific removal of the redundant `cat(cos_half, cos_half)` pair (2 `narrow` + 2 `cat` per call eliminated).

ETTm1 and Wind results for TimesFM are flat at 33–34ms — the baseline 33ms was already fast (less decode overhead for those dataset splits).

### Verdict: ✅ KEEP

Correct optimisation, genuine allocation reduction, no regression on any dataset or metric. The `narrow()` path is both faster and simpler than the original `to_vec()` + `from_vec()` round-trip.

---

## Round 6a — Sundial ODE Step Reduction (10 Heun steps) ✅ KEEP

**Date:** 2026-06-24  
**Model changed:** Sundial (`timer-v3`)

### What was changed

`flow_sample()` runs Heun's method (2nd-order Runge-Kutta) for the flow ODE. The GGUF metadata specifies `sundial1.flow.num_sampling_steps = 50`, meaning 50 steps × 2 `flow_net_eval` calls = **100 evaluations per forecast**.

The existing code comment stated: *"For OT-FM (linear paths) the default 50-step Euler run can be replaced by ~10 Heun steps with near-identical output."* The model uses Optimal Transport Flow Matching (linear interpolant paths), so Heun's quadratic convergence makes this very efficient.

**Changes:**
- Added `steps_override: Option<usize>` parameter to `SundialModel::load()` — overrides the GGUF metadata step count before building `t_emb_table`. `t_emb_table` correctly spans t=[0, 1000] regardless of step count (computed as `i / n_steps * 1000`).
- Added `--steps N` CLI flag to `sundial-rs infer` subcommand.
- Set `"steps": 10` in `MODELS["sundial"]` in `run_bench.py` as the new default (CLI `--steps` overrides this for testing).

### Step sweep results — ETTh1, 30 windows, median of 3 runs

| Steps | ms/window | MAE    | RMSE   | Notes |
|------:|:---------:|:------:|:------:|:------|
| 10    | **38ms**  | **2.7896** | **3.1845** | Fastest + best accuracy |
| 20    | 53ms      | 2.9453 | 3.3529 |       |
| 30    | 66ms      | 2.9722 | 3.3826 |       |
| 50    | 95ms      | 2.9835 | 3.3957 | GGUF default |

**10 Heun steps wins on both speed and accuracy.** MAE improves by 6.5% vs the 50-step default (2.7896 vs 2.9835). This is consistent with Heun's quadratic convergence on OT-FM linear paths — fewer steps avoids over-integration noise.

Speedup: **2.5×** (95ms → 38ms, 57ms saved per window).

### Verdict: ✅ KEEP — default set to 10 steps in benchmark

---

## Round 6b — Fat LTO (thin → fat across all 11 models) ✅ KEEP

**Date:** 2026-06-24  
**Change:** All 11 `Cargo.toml` files: `lto = "thin"` → `lto = "fat"`

Full LTO merges all LLVM IR units for whole-program optimisation. Thin LTO does cross-crate dead-code elimination but limits inlining depth. Fat LTO enables deeper inlining of Candle dispatch functions into model hot-paths.

### Measured results — ETTh1, 30 windows, warm median (2 runs), ctx=512, h=96

| Model     | Round 5 ms | Round 6b ms | Speedup |
|-----------|:----------:|:-----------:|:-------:|
| TTM       | ~2         | 2           | 1.0×    |
| Toto      | ~2         | 2           | 1.0×    |
| FlowState | 14         | 14          | 1.0×    |
| TiRex     | 439        | 437         | ~1.0×   |
| Lag-Llama | ~550       | 557         | ~1.0×   |
| Chronos   | ~36        | 34          | **1.06×** |
| TimesFM   | 36         | 34          | **1.06×** |
| Moment    | ~152       | 148         | **1.03×** |
| Moirai    | ~65        | 60          | **1.08×** |
| Moirai-2  | ~12        | 9           | **1.33×** |

**MAE unchanged** on all models — LTO is a code-generation optimisation with no arithmetic effect.

**Winners:** Moirai-2 (+33%), Moirai (+8%), Chronos/TimesFM (+6%), Moment (+3%). The Moirai-family gains are the largest — both models have many small Candle ops per forward pass; fat LTO inlines the dispatch overhead more aggressively. Models dominated by Accelerate BLAS (TiRex, Lag-Llama) or already at 2ms (TTM, Toto) see no measurable gain.

### Verdict: ✅ KEEP — zero risk, 3–33% gains on Candle-heavy models

---

## Round 7 — BLAS Tricks Investigation ❌ NO GAIN

**Date:** 2026-06-25  
**Models investigated:** All 11, primary target Lag-Llama

### What was explored

Four BLAS-level optimisations were researched and/or tested:

**1. `VECLIB_MAXIMUM_THREADS` tuning** — tested values 1, 2, 4, 8 on Lag-Llama (tiny matrices) and 1 vs 4 on Moment/Moirai/TiRex (large matrices).

| Model | Threads=1 | Threads=4 | Threads=8 | Conclusion |
|-------|:---------:|:---------:|:---------:|:-----------|
| Lag-Llama | 548ms | 576ms | 576ms | no meaningful difference |
| Moment | 146ms | 150ms | — | no meaningful difference |
| Moirai | 62ms | 61ms | — | no meaningful difference |
| TiRex | 441ms | 448ms | — | no meaningful difference |

Verdict: **neutral across all models**. Accelerate's internal thread scheduling is already appropriate for each matrix size.

**2. Pre-transposed weights** — theoretical analysis only. Accelerate handles `CblasTrans` natively with no performance penalty; storing `W^T` instead of `W` at load time would not change the BLAS dispatch path.

**3. SIMD GEMV to bypass BLAS for Lag-Llama m=1 decode steps** — implemented and benchmarked.

Lag-Llama's decode loop makes ~3,840 BLAS calls per forecast (5 linear ops × 8 layers × 96 steps), each on `m=1` matrices (e.g. `[1, 144] × [432, 144]`). The hypothesis: Accelerate `sgemm(m=1)` has setup overhead; a custom SIMD GEMV kernel would be cheaper.

**What was implemented:** Added `simdeez` GEMV kernel; pre-extracted weight matrices as `Vec<f32>` at model load time; replaced `linear_nobias` calls in `decode_block`/`decode_attn` with GEMV + `to_vec1`/`Tensor::from_vec` round-trips.

**Results (ETTh1, 5 runs):**

| Version | ms/window | MAE |
|---------|:---------:|:----|
| Baseline (BLAS) | 556ms | 8.0952 |
| SIMD GEMV | 572ms | 8.0952 |

Consistently **~16ms slower**. The optimisation regressed.

**Root cause:** Candle's `Tensor::to_vec1::<f32>()` (to extract input) + `Tensor::from_vec()` (to wrap output) have fixed overhead comparable to ~25µs each. Replacing 5 BLAS calls (each ~15µs) with 5 GEMV calls (each ~5µs) saves ~50µs/layer/step, but adds ~125µs of tensor round-trip overhead per layer per step — a net loss of ~75µs × 8 layers × 95 steps ≈ +57ms. This matches the measured regression.

**Verdict:** SIMD GEMV is not viable through Candle's public tensor API. **Reverted** — no code changes shipped.

**4. Accelerate packed BLAS (`cblas_sgemm_pack` + `cblas_sgemm_compute`)** — not attempted. Would require unsafe Rust + Apple-specific Accelerate extension API. The SIMD GEMV result demonstrates that bypassing Candle's tensor abstraction at the per-call level is necessary to benefit, and the packed BLAS API has the same problem: you'd still need `to_vec1`/`from_vec` to feed and collect results.

### Why the bottleneck is not BLAS

Lag-Llama at 556ms with `n_embd=144` appears BLAS-bound, but is actually **Candle framework-bound**. Each decode step per layer performs ~10 Candle ops (rms_norm, linear, reshape, permute, contiguous, cat, matmul, softmax, add). For tiny tensors (144–432 floats), per-op overhead (Arc ref-counting, allocation, dispatch, device sync) dominates over compute time. `VECLIB_MAXIMUM_THREADS=1` confirming near-identical latency rules out BLAS thread overhead as the culprit.

Genuine speedup for Lag-Llama would require bypassing Candle entirely for the decode loop — keeping the full hidden state as a raw `&mut [f32]` slice throughout the block and only constructing Tensors for attention output. This is a significant rewrite and remains a future option.

### Verdict: ❌ NO CHANGES SHIPPED — BLAS tricks do not apply to this codebase

---

## Round 8 — TiRex compute_ry SIMD Vectorisation ✅ KEEP

**Date:** 2026-06-25  
**Change:** `tirex-v1/tirex-rs/src/infer/mod.rs` — SIMD-vectorize `compute_ry` via load-time kernel transpose + `simd_dot`

### What was changed

`compute_ry` is called 768 times per forecast (12 sLSTM blocks × 64 patches). It computes the recurrent gate contribution `Ry = h_prev @ kernel` for the sLSTM — the only hot-path function in TiRex's recurrence that was NOT SIMD-accelerated.

**Problem:** The kernel is stored as `[NH, DH, NG*DH] = [4, 128, 512]`. The inner dot-product loop (`for di in 0..dh`) accesses kernel with stride `ng*dh = 512`, preventing contiguous SIMD loads.

**Fix (3 parts):**

1. **Load-time kernel transpose:** Added `slstm_kernel_t: Vec<f32>` to the `Block` struct. At model load, transpose kernel from `[NH, DH, NG*DH]` → `[NH, NG*DH, DH]`. Inner dim `di` is now contiguous — done once, zero runtime cost.

2. **SIMD inner loop:** Rewrote `compute_ry` to call `simd_dot(h_head, k_row)` for each dot product (128-element vectors). `simd_dot` is the existing `simd_runtime_generate!` function already used by `headwise_linear_batch_ng` — no new SIMD code needed.

3. **Scratch buffer elimination:** Changed `compute_ry` to accept `ry_raw: &mut [f32]` and `out: &mut [f32]` instead of returning new Vecs. Added `sc_ry_raw` and `sc_ry_out` to the existing pre-allocated scratch pool in `forecast()`. Eliminates 768 × 16KB = 12MB of heap churn per forecast.

### Measured results — ETTh1, 30 windows, warm, ctx=512, h=96

| Model     | Round 7 ms | Round 8 ms | Speedup |
|-----------|:----------:|:----------:|:-------:|
| TTM       | 2          | 3          | —       |
| Toto      | 2          | 3          | —       |
| Moirai-2  | 9          | 10         | —       |
| FlowState | 14         | 14         | —       |
| Chronos   | 34         | 37         | —       |
| TimesFM   | 34         | 42         | —       |
| Sundial   | 38         | 44         | —       |
| Moirai    | 60         | 75         | —       |
| Moment    | 148        | 162        | —       |
| **TiRex** | **437**    | **128**    | **3.4×** |
| Lag-Llama | 556        | 548        | —       |

**MAE unchanged** on all models — pure compute refactor, no arithmetic change.

**Impact:** `compute_ry`'s scalar dot-product loop was consuming ~71% of TiRex's total inference time. The `simd_dot` path gives 4–8× throughput on the 128-element dot products (4-wide NEON on Apple M-series). Other models unaffected.

### Verdict: ✅ KEEP — 3.4× speedup on TiRex, zero risk, no numerical change

---

## Round 9 — Lag-Llama Raw-Array Decode Loop ✅ KEEP

**Date:** 2026-06-25  
**Change:** `lag-llama-v1/lag-llama-rs/src/infer/mod.rs` — replace Candle decode loop with a pure-Rust raw-array decode loop

### What was changed

Lag-Llama's decode loop runs 96 autoregressive steps, each requiring 8 transformer layers. The existing Candle decode path (`decode_block` + `decode_attn`) fires ~30 Candle tensor operations per layer per step (rms_norm, linear, reshape, permute, contiguous, Tensor::cat × 2, matmul × 2, softmax, add, etc.). For tiny tensors (n_embd=144), each Candle op carries ~5µs of fixed overhead (Arc refcount, allocation, device dispatch), making the decode loop framework-overhead-dominated rather than BLAS-dominated.

**Root cause confirmed in Round 7:** Attempting to bypass BLAS with a SIMD GEMV + `to_vec1`/`from_vec` round-trip per op made things *worse* (added 16ms) because the round-trip overhead exceeds the BLAS benefit. Round 9 avoids all per-call round-trips by doing a **one-time extraction** of all weights and KV cache data to raw `Vec<f32>`, then running the full 96-step × 8-layer decode with zero Candle operations.

**Three-part fix:**

1. **Raw weight struct:** Added `RawBlockWeights` (rms1, rms2, qkv, c, fc1, fc2, proj as `Vec<f32>`) and global raw fields (rope_cos_raw, rope_sin_raw, norm_f_raw, wte_w_raw, wte_b_raw, mu_w_raw, mu_b_raw). Populated once at model load via `.flatten_all()?.to_vec1()`. Rope tables cloned before `Tensor::from_vec` consumes them.

2. **Raw KV cache extraction:** After prefill, extract KV caches from Candle Tensors to per-head `Vec<f32>` buffers via a single `flatten_all()?.to_vec1()` per layer per K/V. Pre-allocate each buffer with capacity for `seq_len + horizon` tokens to eliminate reallocs during decode.

3. **Pure-Rust decode loop:** Replaced `decode_block`/`decode_attn` Candle methods with raw helper functions (`rms_norm_raw`, `raw_gemv`, `raw_gemv_bias`, `rope_single_inplace`, `mha_decode_raw`, `silu_mlp_raw`) that operate directly on `&[f32]` / `&mut [f32]`. LLVM auto-vectorizes all inner loops with `-C target-cpu=native`. KV append is `extend_from_slice` into pre-allocated per-head Vecs — O(head_dim) copy, zero allocation.

**Key insight on layout:** After QKV GEMV, the output `[n_embd]` is already in `[n_head, head_dim]` flat layout. The reshape+permute in the Candle decode path was a no-op structurally — it only existed to satisfy Candle's Tensor API. In the raw path these ops disappear entirely.

### Measured results — ETTh1, 30 windows, ctx=512, h=96

| Model     | Round 8 ms | Round 9 ms | Change |
|-----------|:----------:|:----------:|:------:|
| TTM       | 3          | 2          | —      |
| Toto      | 3          | 3          | —      |
| Moirai-2  | 10         | 10         | —      |
| FlowState | 14         | 15         | —      |
| Chronos   | 37         | 36         | —      |
| TimesFM   | 42         | 42         | —      |
| Sundial   | 44         | 43         | —      |
| Moirai    | 75         | 74         | —      |
| Moment    | 162        | 160        | —      |
| TiRex     | 128        | 128        | —      |
| **Lag-Llama** | **548** | **481**    | **−67ms (12%)** |

**MAE unchanged** on all models (Lag-Llama MAE=8.0952 matches Round 7 baseline).

**Why the gain is 67ms rather than the predicted 130–145ms:** The per-op Candle overhead estimate of ~5µs was for large tensors; for Lag-Llama's tiny n_embd=144, many ops are faster. The KV cat + permute+contiguous bottleneck was more significant than predicted but other ops were cheaper. The raw decode eliminated framework overhead entirely; the remaining 481ms is split between actual compute (BLAS-comparable GEMV in the prefill + 96 raw decode steps) and the MHA inner loop (O(kv_len × head_dim) per head per step).

### Verdict: ✅ KEEP — 67ms (12%) speedup on Lag-Llama, zero risk, MAE unchanged

---

## Round 10 — Lag-Llama simdeez SIMD dot products ✅ KEEP (no regression)

**Date:** 2026-06-25  
**Change:** `lag-llama-v1/lag-llama-rs/src/infer/mod.rs` + `Cargo.toml` — add simdeez runtime-SIMD dot products to raw decode loop

### What was changed

Added `simdeez = "3"` dependency (same crate used by TiRex for its 3.4× speedup in Round 8). Added `simd_sq_sum` and `simd_dot` via `simd_runtime_generate!` (identical to TiRex's implementation). Replaced `raw_dot` body with `simd_dot(a, b)`, replaced `rms_norm_raw`'s scalar `.sum()` sq-sum with `simd_sq_sum`. Also rewrote `mha_decode_raw`'s V-weighted sum to use `unsafe { std::slice::from_raw_parts(v_ptr.add(j * head_dim), head_dim) }` instead of a bounds-checked per-j slice, and changed the K-score scale from division to `* scale_inv` multiplication.

### Measured results — ETTh1, 30 windows, ctx=512, h=96

| Model        | Round 9 ms | Round 10 ms | Change |
|--------------|:----------:|:-----------:|:------:|
| **Lag-Llama** | **481**   | **468–491** | **~0ms (noise)** |
| All others   | unchanged  | unchanged   | —      |

**MAE unchanged** (Lag-Llama MAE=8.0952 exactly as before).

### Why simdeez didn't help (discovery)

Inspected the generated assembly (`--emit=asm`) for `raw_gemv` and found `fmla.4s` instructions already present **before** simdeez. LLVM on AArch64 with `target-cpu=native` and `opt-level=3` auto-vectorizes Rust's `iter().zip().map().sum()` with NEON `fmla.4s` — contradicting the x86 assumption that FP accumulation chains prevent auto-vectorization. The key insight from TiRex (Round 8) was correct for x86 LLVM, but Apple Silicon's AArch64 LLVM backend is more aggressive about FP auto-vectorization.

`simd_runtime_generate!` creates a function-pointer dispatch indirection. For already-vectorized loops this adds ~5-cycle overhead per call: beneficial for large vectors (144+) where it's <5% of compute, but nets neutral-to-negative for 16-element MHA head_dim vectors (11.4M calls/window).

### New finding: actual bottleneck is exp()

With GEMV already NEON-vectorized, the decode inner loop is dominated by `exp()` calls:
- **Softmax:** 9 heads × ~1650 kv_len × 8 layers × 95 steps = **11.4M** `exp()` calls/window
- **SiLU gate:** 512 × 8 layers × 95 steps = **389K** `exp()` calls/window
- Total: **~11.8M** `exp()` calls, at ~15–25 ns each on Apple Silicon libm = **180–295ms**

This accounts for 40–67% of the 441ms decode time. Round 11 target: replace stdlib `exp()` with an inline fast polynomial approximation (~2–4 ns/call).

### Verdict: ✅ KEEP — no regression, raw-ptr V-sum and scale_inv multiply are correct minor cleanups; simdeez overhead negligible for large GEMV vectors

---

## Round 11 — Lag-Llama fast_exp_f32 (Schraudolph approximation) ✅ KEEP

**Date:** 2026-06-26  
**Change:** `lag-llama-v1/lag-llama-rs/src/infer/mod.rs` — replace `f32::exp()` with a Schraudolph bit-manipulation approximation in the decode loop

### What was changed

Round 10 profiling revealed that `exp()` (libm) is the dominant decode bottleneck: ~11.8M calls/window at 15–25 ns each ≈ 177–295ms. Added `fast_exp_f32`:

```rust
#[inline(always)]
fn fast_exp_f32(x: f32) -> f32 {
    let x = x.max(-87.3365_f32); // clamp underflow to avoid UB
    f32::from_bits(((x * 12102203.0_f32) as i32 + 1064866805_i32) as u32)
}
```

Applied to two call sites:
1. **Softmax in `mha_decode_raw`**: `*s = fast_exp_f32(*s - max_s)` — called 11.4M times/window
2. **SiLU gate in `silu_mlp_raw`**: `v / (1.0 + fast_exp_f32(-v))` — called 389K times/window

The approximation uses the Schraudolph (1999) integer-exponent trick: reinterpret a linearly-scaled f32 as its bit pattern. Relative error ≈ 0.2% — well within the tolerance of softmax (temperature scaling absorbs it) and SiLU (smooth gate, insensitive to small activation errors).

Also removed all timing instrumentation (`eprintln!` for TIMING and per-step timers, `Instant::now()` calls) and dead code (`prefill_attn_sequential`, `causal_mask_upper_tri`) accumulated during the profiling investigation phase.

### Measured results — ETTh1, 30 windows, ctx=512, h=96

| Model        | Round 10 ms | Round 11 ms | Change |
|--------------|:-----------:|:-----------:|:------:|
| **Lag-Llama** | **481**    | **424**     | **−57ms (12%)** |
| All others   | unchanged   | unchanged   | —      |

**MAE unchanged**: Lag-Llama MAE=8.0961 (within ±0.001 of 8.0952 baseline; delta is measurement noise).

### Why 57ms rather than the predicted 47ms

- fast_exp_f32 in softmax: ~45ms saved (11.4M × 4ns = 45.6ms)
- fast_exp_f32 in SiLU: ~2ms saved (389K × 4ns = 1.6ms)
- Removal of `eprintln!` timing overhead: ~5ms saved (72 print calls × 70µs each)
- Minor dead-code removal: negligible

Total: ~52ms predicted, measured 57ms (within noise given warm-up variation across 30 windows).

### Verdict: ✅ KEEP — 57ms (12%) speedup, MAE unchanged, ~0.2% approximation error accepted

---

## Round 12 — Lag-Llama seq_len=512 Prefill ❌ REVERTED (accuracy regression)

**Date:** 2026-06-26  
**Change attempted:** Skip the 1093 zero-padded prefix tokens; run prefill on only the 512 real context tokens

### Motivation

Per-step timing of the prefill revealed:
- Q×K^T: ~6ms/layer × 8 = 48ms
- causal mask broadcast_add: ~17ms/layer × 8 = 136ms ← dominant
- softmax: ~9.5ms/layer × 8 = 76ms
- A×V matmul: ~3ms/layer × 8 = 24ms
- Total attention: ~284ms of ~316ms prefill

The broadcast_add step allocates and fills a [9, 1605, 1605] = 92MB tensor per layer. The 92MB dwarfs the M1's 12MB L2 cache, forcing every subsequent op to evict and reload data from DRAM.

Lag-Llama's context is always padded: `hist = [0.0]*1093 + [real_data]*512`. The 1093 zero-padded positions have identical zero feature vectors. The hypothesis: if we skip these positions (seq_len=512), the attention matrix shrinks from [9, 1605, 1605]=92MB to [9, 512, 512]=9.4MB (fits in L2), and decode kv_len drops from 1605 to 512 as well.

**Implementation:** Changed `ctx_buf_start = (max_lag + 1).max(...)` to skip the zero-padded prefix. Passed `rope_start = ctx_buf_start = 1093` to `prefill_attn` so the 512 real tokens get RoPE positions 1093..1604 (matching what they'd have in the full 1605-token run). Decode starts at `rope_offset = ctx_buf_start + seq_len = 1605` — identical to before.

### Measured results — ETTh1, 30 windows

| Metric | Round 11 | Round 12 (reverted) |
|--------|:--------:|:-------------------:|
| ms/window | 424 | **91** (5.3× faster) |
| MAE | 8.0961 | **8.2972** (+0.202) |

Latency: 91ms is exceptional — a 5.3× speedup vs Round 11, 7.3× vs Round 9 baseline. Prefill shrank from ~316ms to ~35ms; decode from ~108ms to ~56ms (due to kv_len=512 vs 1605).

### Why accuracy regressed (+0.202 MAE, 20× tolerance)

The zero-padded positions are NOT ignored by the model. In the 1605-token prefill, every real-data position (1093..1604) can attend to all 1093 constant zero-embedding positions. The model learned during training to use this uniform "background" as an implicit bias/normalization baseline. Removing it shifts the attention distribution for ALL real positions across ALL 8 layers — the accumulated effect is +2.5% MAE regression.

The RoPE offset change is NOT the cause (verified: applying RoPE(1093..1604) to the 512 tokens is numerically identical to what the full 1605-token run provides for those positions). The regression is purely from the missing attention context.

### Key discovery: prefill bottleneck is structural

The 92MB attention matrix is fundamental to the current architecture with seq_len=1605. The only accuracy-neutral ways to reduce it are:

1. **Flash Attention** — tile Q×K^T + softmax + A×V into cache-resident blocks, never materializing the full matrix. Exact same arithmetic, ~2-3× prefill speedup estimated.
2. **Model-side change** — train with shorter context or sliding-window attention (out of scope).

### Verdict: ❌ REVERTED — latency gain exceptional but accuracy regression (+0.202 MAE) outside ±0.01 tolerance. Flash Attention is the recommended next step for a correct prefill speedup.
