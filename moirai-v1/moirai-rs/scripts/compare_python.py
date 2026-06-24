#!/usr/bin/env python3
"""Compare moirai-rs forecast against a pure-Python reference.

The Python reference loads weights directly from models/model.safetensors and
implements the same forward pass as the Rust code (no uni2ts package needed).

Usage:
    python scripts/compare_python.py [--gguf gguf/moirai-f32.gguf] [--horizon 96]
"""
import argparse
import math
import subprocess
import sys

import numpy as np

RUST_BIN = "./target/release/moirai-rs"
SAFETENSORS_PATH = "./models/model.safetensors"

# Model hyperparams (Moirai-1.0-R-large)
PATCH_SIZE  = 32
PATCH_IDX   = 2   # index into [8, 16, 32, 64, 128]
MAX_PS      = 128
D_MODEL     = 1024
N_HEADS     = 16
HEAD_DIM    = 64
N_LAYERS    = 24
MAX_SEQ_LEN = 512


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
# Pure-Python Moirai forward pass
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def qk_norm_heads(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: [seq, n_heads, head_dim] → per-head RMSNorm."""
    seq, nh, hd = x.shape
    x_flat = x.reshape(seq * nh, hd)
    normed = rms_norm(x_flat, w)
    return normed.reshape(seq, nh, hd)


def apply_rope(x: np.ndarray) -> np.ndarray:
    """x: [n_head, seq, head_dim]"""
    n_head, seq, hd = x.shape
    half = hd // 2
    inv_freq = 1.0 / (10000.0 ** (2.0 * np.arange(half, dtype=np.float32) / hd))
    positions = np.arange(seq, dtype=np.float32)
    theta = np.outer(positions, inv_freq)   # [seq, half]
    cos = np.cos(theta)[None, :, :]         # [1, seq, half]
    sin = np.sin(theta)[None, :, :]
    x1, x2 = x[:, :, :half], x[:, :, half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def run_python(context: list[float], horizon: int) -> np.ndarray | None:
    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors not installed — pip install safetensors", file=sys.stderr)
        return None

    w = {}
    try:
        with safe_open(SAFETENSORS_PATH, framework="pt") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).float().numpy()
    except Exception as e:
        print(f"Failed to load safetensors: {e}", file=sys.stderr)
        return None

    # Mean-scale normalization
    ctx = np.array(context, dtype=np.float32)
    loc = ctx.mean()
    scale = max(np.abs(ctx - loc).mean(), 1e-8)

    ctx_norm = (ctx - loc) / scale
    if len(ctx_norm) > MAX_SEQ_LEN:
        ctx_norm = ctx_norm[-MAX_SEQ_LEN:]

    # Pad to multiple of PATCH_SIZE
    rem = len(ctx_norm) % PATCH_SIZE
    if rem != 0:
        ctx_norm = np.concatenate([np.zeros(PATCH_SIZE - rem, dtype=np.float32), ctx_norm])
    n_ctx = len(ctx_norm) // PATCH_SIZE

    n_fc = (horizon + PATCH_SIZE - 1) // PATCH_SIZE
    total = n_ctx + n_fc

    # in_proj: [5, 1024, 128] → take slice [PATCH_IDX, :, :PATCH_SIZE] = [1024, 32]
    proj_w = w["in_proj.weight"][PATCH_IDX, :, :PATCH_SIZE]   # [D_MODEL, PATCH_SIZE]
    proj_b = w["in_proj.bias"][PATCH_IDX]                     # [D_MODEL]

    # Context patch embeddings
    ctx_patches = ctx_norm.reshape(n_ctx, PATCH_SIZE)          # [n_ctx, 32]
    h_ctx = ctx_patches @ proj_w.T + proj_b                    # [n_ctx, D_MODEL]

    # Mask embeddings for future patches
    mask_emb = w["mask_encoding.weight"][0]                    # [D_MODEL]
    h_fc = np.tile(mask_emb[None, :], (n_fc, 1))              # [n_fc, D_MODEL]

    h = np.concatenate([h_ctx, h_fc], axis=0)                 # [total, D_MODEL]

    # is_masked: 0 for context, 1 for future
    is_masked = np.array([0] * n_ctx + [1] * n_fc, dtype=np.int32)

    # Encoder blocks
    for n in range(N_LAYERS):
        p = f"encoder.layers.{n}"
        norm1_w = w[f"{p}.norm1.weight"]
        norm2_w = w[f"{p}.norm2.weight"]
        q_w     = w[f"{p}.self_attn.q_proj.weight"]          # [1024, 1024]
        k_w     = w[f"{p}.self_attn.k_proj.weight"]
        v_w     = w[f"{p}.self_attn.v_proj.weight"]
        o_w     = w[f"{p}.self_attn.out_proj.weight"]
        qn_w    = w[f"{p}.self_attn.q_norm.weight"]          # [64]
        kn_w    = w[f"{p}.self_attn.k_norm.weight"]          # [64]
        vbias_w = w[f"{p}.self_attn.var_attn_bias.emb.weight"]  # [2, 16]
        fc1_w   = w[f"{p}.ffn.fc1.weight"]                   # [2736, 1024]
        fc2_w   = w[f"{p}.ffn.fc2.weight"]                   # [1024, 2736]
        gate_w  = w[f"{p}.ffn.fc_gate.weight"]               # [2736, 1024]

        # Pre-norm attention
        h_norm = rms_norm(h, norm1_w)
        q = h_norm @ q_w.T   # [total, D_MODEL]
        k = h_norm @ k_w.T
        v = h_norm @ v_w.T

        q = q.reshape(total, N_HEADS, HEAD_DIM)
        k = k.reshape(total, N_HEADS, HEAD_DIM)

        q = qk_norm_heads(q, qn_w)
        k = qk_norm_heads(k, kn_w)

        # [total, N_HEADS, HEAD_DIM] → [N_HEADS, total, HEAD_DIM]
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.reshape(total, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        scale_attn = math.sqrt(HEAD_DIM)
        scores = np.einsum("hqd,hkd->hqk", q, k) / scale_attn  # [N_HEADS, total, total]

        # Variate attention bias: vbias_w [2, N_HEADS]; bias per key based on is_masked
        # Python shape [2, N_HEADS]: vbias_w[type, head]
        for key_idx in range(total):
            bias_type = int(is_masked[key_idx])
            for head in range(N_HEADS):
                b = vbias_w[bias_type, head]
                scores[head, :, key_idx] += b

        # Softmax
        scores = np.where(np.isfinite(scores), scores, -1e9)
        attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

        out = np.einsum("hqs,hsd->hqd", attn_weights, v)  # [N_HEADS, total, HEAD_DIM]
        out = out.transpose(1, 0, 2).reshape(total, D_MODEL)  # [total, D_MODEL]
        out = out @ o_w.T
        h = h + out

        # FFN (SwiGLU)
        h_norm2 = rms_norm(h, norm2_w)
        content = silu(h_norm2 @ fc1_w.T)       # [total, 2736]
        gate    = h_norm2 @ gate_w.T             # [total, 2736]
        h = h + (content * gate) @ fc2_w.T

    # Final norm
    h = rms_norm(h, w["encoder.norm.weight"])

    # Head: st_loc [5, 128, 1024] → slice [PATCH_IDX, :PATCH_SIZE, :] = [32, 1024]
    loc_w = w["param_proj.proj.components.0.loc.weight"][PATCH_IDX, :PATCH_SIZE, :]  # [32, 1024]
    loc_b = w["param_proj.proj.components.0.loc.bias"][PATCH_IDX, :PATCH_SIZE]       # [32]

    future_h = h[n_ctx:n_ctx + n_fc]   # [n_fc, D_MODEL]
    pred = future_h @ loc_w.T + loc_b  # [n_fc, 32]

    pred_flat = pred.flatten()[:horizon]
    return (pred_flat * scale + loc).astype(np.float32)


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
    parser.add_argument("--gguf", default="gguf/moirai-f32.gguf")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    context = make_context(args.context_len)
    print(f"Context: {len(context)} steps  |  Horizon: {args.horizon}")

    print(f"\n--- Running Rust ({args.gguf}) ---")
    rust_out = run_rust(args.gguf, context, args.horizon)
    print(f"  Rust shape: {rust_out.shape}  first 5: {rust_out[:5]}")

    print("\n--- Running Python (Moirai, safetensors) ---")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("\nRust output (first 10):")
        print(rust_out[:10])
        return

    print(f"  Python shape: {py_out.shape}  first 5: {py_out[:5]}")
    report(rust_out, py_out, args.horizon)


if __name__ == "__main__":
    main()
