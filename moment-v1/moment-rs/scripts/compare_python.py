#!/usr/bin/env python3
"""Compare moment-rs forecast against the Python momentfm reference.

Both engines load weights from the same local models/ directory.
Input: fixed 512-step sine+trend series.  Horizon: 96 steps.

Python reference requires:
    pip install momentfm torch

Usage:
    python scripts/compare_python.py [--gguf moment.gguf] [--horizon 96]
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np

RUST_BIN = "./target/release/moment-rs"
MODEL_DIR = "./models"


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


def run_python(context: list[float], horizon: int) -> np.ndarray | None:
    """
    Replicate the Rust zero-shot masked-inpainting approach:
      - Load model with task_name='reconstruction' (keeps checkpoint head weights)
      - RevIN normalize the context
      - Patch context (64 patches of 8) + zero future patches
      - Run encoder, apply head to future patch positions
      - RevIN denormalize
    """
    try:
        import torch
        from momentfm import MOMENTPipeline
    except ImportError:
        print("momentfm not installed — pip install momentfm torch", file=sys.stderr)
        return None

    PATCH_LEN = 8
    SEQ_LEN = 512
    CTX_PATCHES = SEQ_LEN // PATCH_LEN  # 64

    try:
        # Load with reconstruction task so checkpoint head is kept (not re-initialized)
        pipe = MOMENTPipeline.from_pretrained(
            os.path.abspath(MODEL_DIR),
            model_kwargs={"task_name": "reconstruction"},
        )
        pipe.init()
        pipe.eval()

        # RevIN stats
        ctx = np.array(context, dtype=np.float32)
        loc = ctx.mean()
        scale = max(ctx.std(), 1e-8)
        ctx_norm = (ctx - loc) / scale

        # Pad/truncate to SEQ_LEN
        if len(ctx_norm) < SEQ_LEN:
            pad = SEQ_LEN - len(ctx_norm)
            ctx_norm = np.concatenate([np.zeros(pad, dtype=np.float32), ctx_norm])
        else:
            ctx_norm = ctx_norm[-SEQ_LEN:]

        # Build patches: [CTX_PATCHES + n_fc, PATCH_LEN]
        n_fc = (horizon + PATCH_LEN - 1) // PATCH_LEN
        patches = [ctx_norm[i*PATCH_LEN:(i+1)*PATCH_LEN] for i in range(CTX_PATCHES)]
        patches += [np.zeros(PATCH_LEN, dtype=np.float32) for _ in range(n_fc)]
        total_patches = CTX_PATCHES + n_fc

        ctx_x = torch.tensor(np.stack(patches[:CTX_PATCHES]), dtype=torch.float32)  # [CTX_PATCHES, 8]

        with torch.no_grad():
            # value embedding for context patches: [CTX_PATCHES, 1024]
            h_ctx = pipe.patch_embedding.value_embedding(ctx_x)
            # mask embedding for future patches (what the model was trained to decode from)
            mask_emb = pipe.patch_embedding.mask_embedding  # [1024]
            h_fc = mask_emb.unsqueeze(0).expand(n_fc, -1)  # [n_fc, 1024]
            h = torch.cat([h_ctx, h_fc], dim=0)  # [total_patches, 1024]
            # position embedding
            pos = pipe.patch_embedding.position_embedding.pe[0, :total_patches, :]
            h = h + pos  # [total_patches, 1024]

            # T5 encoder expects [batch, seq, d_model]
            h = h.unsqueeze(0)  # [1, total_patches, 1024]
            enc_out = pipe.encoder(inputs_embeds=h)
            h = enc_out.last_hidden_state.squeeze(0)  # [total_patches, 1024]

            # Apply head to future positions
            # Head expects [batch, n_patches, n_vars, d_model] → [batch, n_patches, n_vars*patch_len]
            future_h = h[CTX_PATCHES:CTX_PATCHES + n_fc]  # [n_fc, 1024]
            future_h = future_h.unsqueeze(0).unsqueeze(2)  # [1, n_fc, 1, 1024]
            pred = pipe.head(future_h)  # [1, n_fc, 8]

        pred_flat = pred.flatten().numpy()
        result = pred_flat[:horizon] * scale + loc
        return result
    except Exception as e:
        print(f"Python inference failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return None


def report(rust: np.ndarray, python: np.ndarray, horizon: int) -> None:
    n = min(len(rust), len(python), horizon)
    r, p = rust[:n], python[:n]
    diff = np.abs(r - p)
    print(f"\nComparison over {n} steps:")
    print(f"  Max  |Δ|: {diff.max():.6f}")
    print(f"  Mean |Δ|: {diff.mean():.6f}")
    print(f"  MAE      : {diff.mean():.6f}")
    print(f"\nFirst 8 steps:")
    print(f"  Python: {' '.join(f'{v:.4f}' for v in p[:8])}")
    print(f"    Rust: {' '.join(f'{v:.4f}' for v in r[:8])}")
    print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a,b in zip(p[:8],r[:8]))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", default="gguf/moment-f32.gguf")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--context-len", type=int, default=512)
    args = parser.parse_args()

    context = make_context(args.context_len)
    print(f"Context: {len(context)} steps  |  Horizon: {args.horizon}")

    print(f"\n--- Running Rust ({args.gguf}) ---")
    rust_out = run_rust(args.gguf, context, args.horizon)
    print(f"  Rust shape: {rust_out.shape}  first 5: {rust_out[:5]}")

    print("\n--- Running Python (momentfm) ---")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("\nRust output (first 10):")
        print(rust_out[:10])
        return

    print(f"  Python shape: {py_out.shape}  first 5: {py_out[:5]}")
    report(rust_out, py_out, args.horizon)


if __name__ == "__main__":
    main()
