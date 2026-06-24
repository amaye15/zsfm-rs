#!/usr/bin/env python3
"""Compare moirai-2-rs forecast against a pure-numpy reference.

The Python reference loads weights directly from models/model.safetensors and
implements the same forward pass as the Rust code (no uni2ts package needed).

Architecture: Moirai-2.0-R-small
  - PackedStdScaler normalization (mean + std, Bessel correction)
  - ResidualBlock in_proj: [patch_size*2=32] → [d_model=384]
  - 6 causal encoder blocks: RMSNorm + QK-norm + partial interleaved RoPE + SwiGLU FFN
  - ResidualBlock out_proj: [384] → [576=4*9*16]
  - Multi-token decode loop: 4 patches per forward pass

Usage:
    python scripts/compare_python.py [--gguf gguf/moirai2-f32.gguf] [--horizon 96]
"""
import argparse
import math
import subprocess
import sys

import numpy as np

RUST_BIN        = "./target/release/moirai-2-rs"
SAFETENSORS_PATH = "./models/model.safetensors"

# Moirai-2.0-R-small hyperparams
D_MODEL         = 384
N_HEADS         = 6
HEAD_DIM        = 64
N_LAYERS        = 6
D_FF            = 1024
PATCH_SIZE      = 16
NUM_PREDICT_TOK = 4
NUM_QUANTILES   = 9
MEDIAN_Q        = 4      # index 4 of 9 = 0.5 quantile
ROPE_DIM        = 32     # partial_factor=(0.0, 0.5) → first 32 of 64 dims
MAX_SEQ_LEN     = 512


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
# Pure-numpy Moirai-2.0 forward pass
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def residual_block(x: np.ndarray, hidden_w, hidden_b, output_w, output_b, residual_w, residual_b):
    """out = output(silu(hidden(x))) + residual(x)"""
    h   = silu(x @ hidden_w.T + hidden_b)
    out = h @ output_w.T + output_b
    res = x @ residual_w.T + residual_b
    return out + res


def qk_norm_heads(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: [seq, n_heads, head_dim] → per-head RMSNorm with shared weight."""
    seq, nh, hd = x.shape
    x_flat = x.reshape(seq * nh, hd)
    normed = rms_norm(x_flat, w)
    return normed.reshape(seq, nh, hd)


def apply_partial_rope(x: np.ndarray, time_ids: list[int]) -> np.ndarray:
    """Partial interleaved RoPE on x [n_heads, seq, head_dim].
    Applies to first rope_dim=32 dims using paired rotation.
    inv_freq[i] = 1 / 10000^(2i / rope_dim) for i in [0, rope_dim/2).
    """
    n_heads, seq, hd = x.shape
    half_rope = ROPE_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (2.0 * np.arange(half_rope, dtype=np.float32) / ROPE_DIM))

    x_out = x.copy()
    for t, pos in enumerate(time_ids):
        for i in range(half_rope):
            theta = pos * inv_freq[i]
            cos_v = np.cos(theta)
            sin_v = np.sin(theta)
            ie = 2 * i
            io = 2 * i + 1
            v_e = x[:, t, ie].copy()
            v_o = x[:, t, io].copy()
            x_out[:, t, ie] = cos_v * v_e - sin_v * v_o
            x_out[:, t, io] = sin_v * v_e + cos_v * v_o
    return x_out


def encoder_block(h, n, w, time_ids):
    """One causal encoder block."""
    seq = len(h)
    p = f"encoder.layers.{n}"

    norm1_w   = w[f"{p}.norm1.weight"]
    norm2_w   = w[f"{p}.norm2.weight"]
    q_w       = w[f"{p}.self_attn.q_proj.weight"]       # [384, 384]
    k_w       = w[f"{p}.self_attn.k_proj.weight"]
    v_w       = w[f"{p}.self_attn.v_proj.weight"]
    o_w       = w[f"{p}.self_attn.out_proj.weight"]
    qn_w      = w[f"{p}.self_attn.q_norm.weight"]       # [64]
    kn_w      = w[f"{p}.self_attn.k_norm.weight"]
    vbias_w   = w[f"{p}.self_attn.var_attn_bias.emb.weight"]  # [2, 6]
    fc1_w     = w[f"{p}.ffn.fc1.weight"]                # [1024, 384]
    fc2_w     = w[f"{p}.ffn.fc2.weight"]                # [384, 1024]
    gate_w    = w[f"{p}.ffn.fc_gate.weight"]            # [1024, 384]

    # Pre-norm attention
    h_norm = rms_norm(h, norm1_w)
    q = h_norm @ q_w.T   # [seq, 384]
    k = h_norm @ k_w.T
    v = h_norm @ v_w.T

    q = q.reshape(seq, N_HEADS, HEAD_DIM)
    k = k.reshape(seq, N_HEADS, HEAD_DIM)

    q = qk_norm_heads(q, qn_w)
    k = qk_norm_heads(k, kn_w)

    # [seq, N_HEADS, HEAD_DIM] → [N_HEADS, seq, HEAD_DIM]
    q = q.transpose(1, 0, 2)
    k = k.transpose(1, 0, 2)
    v = v.reshape(seq, N_HEADS, HEAD_DIM).transpose(1, 0, 2)

    # Partial interleaved RoPE
    q = apply_partial_rope(q, time_ids)
    k = apply_partial_rope(k, time_ids)

    scale_attn = math.sqrt(HEAD_DIM)
    scores = np.einsum("hqd,hkd->hqk", q, k) / scale_attn  # [N_HEADS, seq, seq]

    # Causal mask: -inf where j > i
    causal = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(i + 1, seq):
            causal[i, j] = -np.inf
    scores = scores + causal[None, :, :]  # broadcast over heads

    # BinaryAttentionBias: univariate → all same variate (type=1)
    # vbias_w[1, head] for each head — constant across all (query, key) pairs
    for head in range(N_HEADS):
        b = vbias_w[1, head]
        scores[head] += b

    # Softmax (handle -inf safely)
    max_s = scores.max(axis=-1, keepdims=True)
    scores_shifted = scores - max_s
    scores_shifted = np.where(np.isfinite(scores_shifted), scores_shifted, -1e9)
    attn_weights = np.exp(scores_shifted)
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

    out = np.einsum("hqs,hsd->hqd", attn_weights, v)   # [N_HEADS, seq, HEAD_DIM]
    out = out.transpose(1, 0, 2).reshape(seq, D_MODEL)
    out = out @ o_w.T
    h = h + out

    # SwiGLU FFN
    h_norm2 = rms_norm(h, norm2_w)
    content = silu(h_norm2 @ fc1_w.T)      # [seq, 1024]
    gate    = h_norm2 @ gate_w.T           # [seq, 1024]
    h = h + (content * gate) @ fc2_w.T

    return h


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

    # --- PackedStdScaler normalization (context only) ---
    ctx = np.array(context, dtype=np.float32)
    max_ctx_len = (MAX_SEQ_LEN // PATCH_SIZE) * PATCH_SIZE
    if len(ctx) > max_ctx_len:
        ctx = ctx[-max_ctx_len:]

    n = len(ctx)
    loc = ctx.mean().astype(np.float64)
    var = ((ctx.astype(np.float64) - loc) ** 2).sum() / max(n - 1, 1)
    scale = float(np.sqrt(var + 1e-5))
    loc   = float(loc)

    ctx_norm = (ctx - loc) / scale

    # Left-pad to multiple of PATCH_SIZE
    rem = len(ctx_norm) % PATCH_SIZE
    if rem != 0:
        ctx_norm = np.concatenate([np.zeros(PATCH_SIZE - rem, dtype=np.float32), ctx_norm])
    n_ctx = len(ctx_norm) // PATCH_SIZE

    # --- Build context input tokens: [norm_values(16), ones(16)] ---
    ctx_patches = ctx_norm.reshape(n_ctx, PATCH_SIZE)
    ones_mask   = np.ones((n_ctx, PATCH_SIZE), dtype=np.float32)
    all_tokens  = np.concatenate([ctx_patches, ones_mask], axis=1)  # [n_ctx, 32]
    time_ids    = list(range(n_ctx))

    # --- Residual block helpers from weight dict ---
    def in_proj_fwd(x):
        return residual_block(
            x,
            w["in_proj.hidden_layer.weight"], w["in_proj.hidden_layer.bias"],
            w["in_proj.output_layer.weight"], w["in_proj.output_layer.bias"],
            w["in_proj.residual_layer.weight"], w["in_proj.residual_layer.bias"],
        )

    def out_proj_fwd(x):
        return residual_block(
            x,
            w["out_proj.hidden_layer.weight"], w["out_proj.hidden_layer.bias"],
            w["out_proj.output_layer.weight"], w["out_proj.output_layer.bias"],
            w["out_proj.residual_layer.weight"], w["out_proj.residual_layer.bias"],
        )

    norm_f_w = w["encoder.norm.weight"]

    # --- Multi-token decode loop ---
    n_future_patches = (horizon + PATCH_SIZE - 1) // PATCH_SIZE
    collected_patches: list[np.ndarray] = []

    while len(collected_patches) < n_future_patches:
        seq_len = len(all_tokens)

        # Forward pass
        h = in_proj_fwd(all_tokens)  # [seq, 384]
        for layer_n in range(N_LAYERS):
            h = encoder_block(h, layer_n, w, time_ids)
        h = rms_norm(h, norm_f_w)
        preds = out_proj_fwd(h)  # [seq, 576]

        # Last token output → [576] → [4, 9, 16]
        last_pred = preds[-1]  # [576]
        reshaped = last_pred.reshape(NUM_PREDICT_TOK, NUM_QUANTILES, PATCH_SIZE)
        pred_patches = reshaped[:, MEDIAN_Q, :]  # [4, 16]

        # Collect up to what we need
        need   = n_future_patches - len(collected_patches)
        n_take = min(NUM_PREDICT_TOK, need)
        for pt in range(n_take):
            collected_patches.append(pred_patches[pt])

        # If more passes needed, append predicted tokens (mask=0)
        if len(collected_patches) < n_future_patches:
            last_time = time_ids[-1]
            for i, patch in enumerate(pred_patches):
                tok = np.concatenate([patch, np.zeros(PATCH_SIZE, dtype=np.float32)])
                all_tokens = np.vstack([all_tokens, tok])
                time_ids.append(last_time + 1 + i)

    # Flatten and denormalize
    pred_flat = np.concatenate(collected_patches)[:horizon]
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
    print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a, b in zip(p[:8], r[:8]))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="gguf/moirai2-f32.gguf")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    context = make_context(args.context_len)
    print(f"Context: {len(context)} steps  |  Horizon: {args.horizon}")

    print(f"\n--- Running Rust ({args.gguf}) ---")
    try:
        rust_out = run_rust(args.gguf, context, args.horizon)
        print(f"  Rust shape: {rust_out.shape}  first 5: {rust_out[:5]}")
    except subprocess.CalledProcessError as e:
        print(f"  Rust failed: {e.stderr}", file=sys.stderr)
        rust_out = None

    print("\n--- Running Python (Moirai-2.0, safetensors) ---")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("Python reference unavailable.")
        if rust_out is not None:
            print("Rust output (first 10):", rust_out[:10])
        return

    print(f"  Python shape: {py_out.shape}  first 5: {py_out[:5]}")

    if rust_out is not None:
        report(rust_out, py_out, args.horizon)
    else:
        print("Python output (first 10):", py_out[:10])


if __name__ == "__main__":
    main()
