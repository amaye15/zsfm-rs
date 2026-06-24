#!/usr/bin/env python3
"""Compare ttm-rs Rust inference vs. Python TinyTimeMixer (granite-timeseries-ttm-r2).

Usage:
    python scripts/compare_python.py [--gguf gguf/ttm-f32.gguf] [--horizon 96]
"""
import argparse
import math
import subprocess
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_BIN = os.path.join(REPO_ROOT, "target", "release", "ttm-rs")
CONFIG    = os.path.join(REPO_ROOT, "models", "config.json")


def make_context(n=512):
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.05 * t + 1.0).astype(np.float32)


def run_rust(gguf: str, context: np.ndarray, horizon: int) -> np.ndarray:
    import json
    request = json.dumps({"context": context.tolist(), "horizon": horizon})
    result = subprocess.run(
        [RUST_BIN, "infer", "--gguf", gguf, "--config", CONFIG],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    return np.array(fc["point"][:horizon], dtype=np.float32)


def run_python(context: np.ndarray, horizon: int) -> np.ndarray:
    try:
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    except ImportError:
        print("tsfm_public not available; skipping Python reference.", file=sys.stderr)
        return None

    try:
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2")
        model.eval()
        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            out = model(past_values=x)
        pred = out.prediction_outputs[0, :horizon, 0].numpy()
        return pred.astype(np.float32)
    except Exception as e:
        print(f"Python model failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="gguf/ttm-f32.gguf")
    parser.add_argument("--horizon", type=int, default=96)
    args = parser.parse_args()

    context = make_context(512)
    print(f"Context length: {len(context)}, horizon: {args.horizon}")

    print(f"Running Rust inference ({args.gguf}) …")
    rs_out = run_rust(args.gguf, context, args.horizon)
    print(f"  Rust output shape: {rs_out.shape}, first 5: {rs_out[:5]}")

    print("Running Python TinyTimeMixer …")
    py_out = run_python(context, args.horizon)
    if py_out is None:
        print("Cannot compare without Python reference.")
        print("\nRust output (first 10):")
        print(rs_out[:10])
        return

    n = min(len(rs_out), len(py_out))
    diff = np.abs(rs_out[:n] - py_out[:n])
    print(f"\nComparison over {n} steps:")
    print(f"  Max  |Δ|: {diff.max():.6f}")
    print(f"  Mean |Δ|: {diff.mean():.6f}")


if __name__ == "__main__":
    main()
