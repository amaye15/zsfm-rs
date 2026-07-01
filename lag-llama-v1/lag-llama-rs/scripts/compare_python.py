#!/usr/bin/env python3
"""Compare lag-llama-rs forecast against a pure-Python reference.

The Python reference loads weights directly from lag_llama.safetensors and
implements the same forward pass as the Rust code (no lag-llama package needed).

Usage:
    python scripts/compare_python.py [--gguf gguf/lag_llama-f32.gguf] [--horizon 96]
"""
import argparse
import math
import subprocess
import sys

import numpy as np

RUST_BIN = "./target/release/lag-llama-rs"
SAFETENSORS_PATH = "./models/lag_llama.safetensors"

LAGS_SEQ = [
    0, 7, 8, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30,
    34, 35, 36, 46, 47, 48, 50, 51, 52, 55, 57, 58, 59, 60, 61, 70, 71, 72,
    83, 94, 95, 96, 102, 103, 104, 117, 118, 119, 120, 121, 142, 143, 144,
    154, 155, 156, 166, 167, 168, 177, 178, 179, 180, 181, 334, 335, 336,
    362, 363, 364, 502, 503, 504, 670, 671, 672, 718, 719, 720, 726, 727,
    728, 1090, 1091, 1092,
]
N_LAGS = len(LAGS_SEQ)       # 84
FEATURE_SIZE = 92             # 84 lags + 8 time features (zeroed)
N_EMBD = 144
N_HEAD = 9
HEAD_DIM = 16                 # N_EMBD // N_HEAD
N_LAYER = 8
MAX_CTX = 2048


def make_context(n: int = 512) -> list[float]:
    return [math.sin(2 * math.pi * i / 48) + 0.01 * i for i in range(n)]


def run_rust(gguf: str, context: list[float], horizon: int) -> np.ndarray:
    import json
    request = json.dumps({"context": list(context), "horizon": horizon})
    result = subprocess.run(
        [RUST_BIN, "infer", "--gguf", gguf],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    return np.array(fc["point"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pure-Python Lag-Llama forward pass
# ---------------------------------------------------------------------------

def robust_stats(x: np.ndarray):
    n = len(x)
    sorted_x = np.sort(x)
    med = (sorted_x[n // 2 - 1] + sorted_x[n // 2]) / 2 if n % 2 == 0 else sorted_x[n // 2]
    devs = np.sort(np.abs(sorted_x - med))
    mad = (devs[n // 2 - 1] + devs[n // 2]) / 2 if n % 2 == 0 else devs[n // 2]
    return med, mad


def build_features(hist: list[float], abs_t: int) -> np.ndarray:
    feat = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for li, lag in enumerate(LAGS_SEQ):
        src = abs_t - lag
        if 0 <= src < len(hist):
            feat[li] = hist[src]
    return feat


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def apply_rope(x: np.ndarray, offset: int) -> np.ndarray:
    """x: [n_head, seq, head_dim]"""
    n_head, seq, hd = x.shape
    half = hd // 2
    inv_freq = 1.0 / (10000.0 ** (2.0 * np.arange(half, dtype=np.float32) / hd))
    positions = np.arange(offset, offset + seq, dtype=np.float32)
    theta = np.outer(positions, inv_freq)  # [seq, half]
    cos = np.cos(theta)[None, :, :]       # [1, seq, half]
    sin = np.sin(theta)[None, :, :]       # [1, seq, half]
    x1 = x[:, :, :half]
    x2 = x[:, :, half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return np.concatenate([rot1, rot2], axis=-1)


def run_python(context: list[float], horizon: int) -> np.ndarray | None:
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        print("torch / safetensors not installed", file=sys.stderr)
        return None

    # Load weights
    w = {}
    try:
        with safe_open(SAFETENSORS_PATH, framework="pt") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).float().numpy()
    except Exception as e:
        print(f"Failed to load safetensors: {e}", file=sys.stderr)
        return None

    max_lag = max(LAGS_SEQ)
    loc, scale = robust_stats(np.array(context, dtype=np.float32))
    scale = max(scale, 1e-8)

    hist = [0.0] * (max_lag + 1) + [(v - loc) / scale for v in context]

    ctx_end = len(hist)
    ctx_start = max(0, ctx_end - MAX_CTX)
    seq_len = ctx_end - ctx_start

    # Build feature matrix [seq_len, FEATURE_SIZE]
    feat = np.stack([build_features(hist, ctx_start + t) for t in range(seq_len)])

    # WTE: linear(feat, wte_w, wte_b) → [seq_len, N_EMBD]
    wte_w = w["model.transformer.wte.weight"]  # [144, 92]
    wte_b = w["model.transformer.wte.bias"]    # [144]
    h = feat @ wte_w.T + wte_b                # [seq_len, 144]

    # Per-layer KV caches: list of (K, V) each [n_head, seq_len, head_dim]
    kv_caches = []

    for n in range(N_LAYER):
        p = f"model.transformer.h.{n}"
        rms1_w = w[f"{p}.rms_1.scale"]
        rms2_w = w[f"{p}.rms_2.scale"]
        q_w    = w[f"{p}.attn.q_proj.weight"]   # [144, 144]
        kv_w   = w[f"{p}.attn.kv_proj.weight"]  # [288, 144]
        c_w    = w[f"{p}.attn.c_proj.weight"]   # [144, 144]
        fc1_w  = w[f"{p}.mlp.c_fc1.weight"]     # [512, 144]
        fc2_w  = w[f"{p}.mlp.c_fc2.weight"]     # [512, 144]
        proj_w = w[f"{p}.mlp.c_proj.weight"]    # [144, 512]

        # Pre-norm attention
        h_norm = rms_norm(h, rms1_w)            # [seq, 144]
        q = h_norm @ q_w.T                      # [seq, 144]
        kv = h_norm @ kv_w.T                    # [seq, 288]
        k = kv[:, :N_EMBD]                      # [seq, 144]
        v = kv[:, N_EMBD:]                      # [seq, 144]

        # Reshape to [n_head, seq, head_dim]
        q = q.reshape(seq_len, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
        k = k.reshape(seq_len, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
        v = v.reshape(seq_len, N_HEAD, HEAD_DIM).transpose(1, 0, 2)

        # RoPE
        q = apply_rope(q, 0)
        k = apply_rope(k, 0)

        # Causal attention
        scale_attn = math.sqrt(HEAD_DIM)
        scores = np.einsum("hqd,hkd->hqk", q, k) / scale_attn  # [n_head, seq, seq]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores += mask[None, :, :]
        scores = np.where(np.isfinite(scores), scores, -1e9)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn /= attn.sum(axis=-1, keepdims=True)
        out = np.einsum("hqs,hsd->hqd", attn, v)  # [n_head, seq, head_dim]
        out = out.transpose(1, 0, 2).reshape(seq_len, N_EMBD)  # [seq, 144]
        out = out @ c_w.T
        h = h + out

        # FFN
        h_norm2 = rms_norm(h, rms2_w)
        gate = silu(h_norm2 @ fc1_w.T)          # [seq, 512]
        val  = h_norm2 @ fc2_w.T                 # [seq, 512]
        h = h + (gate * val) @ proj_w.T

        kv_caches.append((k, v))

    # Final norm + head on last context token
    norm_f_w = w["model.transformer.ln_f.scale"]
    mu_w = w["model.param_proj.proj.0.weight"]  # [1, 144]
    mu_b = w["model.param_proj.proj.0.bias"]    # [1]

    preds = []
    last_h = h[-1]  # [144]
    rope_offset = seq_len

    for step in range(horizon):
        normed = rms_norm(last_h, norm_f_w)
        mu_val = (normed @ mu_w.T + mu_b).item()
        preds.append(mu_val * scale + loc)

        if step == horizon - 1:
            break

        # Append to history and build next token features
        hist.append(mu_val)
        abs_t = len(hist) - 1
        feat1 = build_features(hist, abs_t).reshape(1, FEATURE_SIZE)

        h1 = feat1 @ wte_w.T + wte_b            # [1, 144]

        for n in range(N_LAYER):
            p = f"model.transformer.h.{n}"
            rms1_w = w[f"{p}.rms_1.scale"]
            rms2_w = w[f"{p}.rms_2.scale"]
            q_w    = w[f"{p}.attn.q_proj.weight"]
            kv_w   = w[f"{p}.attn.kv_proj.weight"]
            c_w    = w[f"{p}.attn.c_proj.weight"]
            fc1_w  = w[f"{p}.mlp.c_fc1.weight"]
            fc2_w  = w[f"{p}.mlp.c_fc2.weight"]
            proj_w = w[f"{p}.mlp.c_proj.weight"]

            h1_norm = rms_norm(h1, rms1_w)
            q1 = h1_norm @ q_w.T                # [1, 144]
            kv1 = h1_norm @ kv_w.T
            k1 = kv1[:, :N_EMBD]
            v1 = kv1[:, N_EMBD:]

            q1 = q1.reshape(1, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
            k1_raw = k1.reshape(1, N_HEAD, HEAD_DIM).transpose(1, 0, 2)
            v1 = v1.reshape(1, N_HEAD, HEAD_DIM).transpose(1, 0, 2)

            q1 = apply_rope(q1, rope_offset)
            k1_rope = apply_rope(k1_raw, rope_offset)

            k_full = np.concatenate([kv_caches[n][0], k1_rope], axis=1)
            v_full = np.concatenate([kv_caches[n][1], v1], axis=1)

            scale_attn = math.sqrt(HEAD_DIM)
            scores1 = np.einsum("hqd,hkd->hqk", q1, k_full) / scale_attn
            attn1 = np.exp(scores1 - scores1.max(axis=-1, keepdims=True))
            attn1 /= attn1.sum(axis=-1, keepdims=True)
            out1 = np.einsum("hqs,hsd->hqd", attn1, v_full)
            out1 = out1.transpose(1, 0, 2).reshape(1, N_EMBD)
            out1 = out1 @ c_w.T
            h1 = h1 + out1

            h1_norm2 = rms_norm(h1, rms2_w)
            gate1 = silu(h1_norm2 @ fc1_w.T)
            val1  = h1_norm2 @ fc2_w.T
            h1 = h1 + (gate1 * val1) @ proj_w.T

            kv_caches[n] = (k_full, v_full)

        last_h = h1[0]
        rope_offset += 1

    return np.array(preds, dtype=np.float32)


def report(rust: np.ndarray, python: np.ndarray, horizon: int) -> None:
    n = min(len(rust), len(python), horizon)
    r, p = rust[:n], python[:n]
    diff = np.abs(r - p)
    print(f"\nComparison over {n} steps:")
    print(f"  Max  |Δ|: {diff.max():.6f}")
    print(f"  Mean |Δ|: {diff.mean():.6f}")
    print(f"\nFirst 8 steps:")
    print(f"  Python: {' '.join(f'{v:.4f}' for v in p[:8])}")
    print(f"    Rust: {' '.join(f'{v:.4f}' for v in r[:8])}")
    print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a,b in zip(p[:8],r[:8]))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="gguf/lag_llama-f32.gguf")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    context = make_context(args.context_len)
    print(f"Context: {len(context)} steps  |  Horizon: {args.horizon}")

    print(f"\n--- Running Rust ({args.gguf}) ---")
    rust_out = run_rust(args.gguf, context, args.horizon)
    print(f"  Rust shape: {rust_out.shape}  first 5: {rust_out[:5]}")

    print("\n--- Running Python (Lag-Llama, safetensors) ---")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("\nRust output (first 10):")
        print(rust_out[:10])
        return

    print(f"  Python shape: {py_out.shape}  first 5: {py_out[:5]}")
    report(rust_out, py_out, args.horizon)


if __name__ == "__main__":
    main()
