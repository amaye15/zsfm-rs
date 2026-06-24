#!/usr/bin/env python3
"""Compare toto-rs Rust inference vs. Python Toto-2.0 reference.

NOTE: The Toto Python package API has diverged from the released weights.
This script runs the Rust binary for all three dtypes and compares f16/q8
against f32 (intra-Rust comparison) to measure quantization degradation.
A full Python vs Rust comparison requires the original toto training codebase.

Usage:
    python scripts/compare_python.py [--model 22m] [--dtype f32|f16|q8]
"""
import argparse
import json
import math
import subprocess
import sys
import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUST_BIN  = os.path.join(REPO_ROOT, "target", "release", "toto-rs")


def make_context(n=128):
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.05 * t + 1.0).astype(np.float32)


def run_rust(model: str, dtype: str, context: np.ndarray, horizon: int = 32) -> np.ndarray:
    gguf   = os.path.join(REPO_ROOT, "gguf", f"toto-{model}-{dtype}.gguf")
    config = os.path.join(REPO_ROOT, "models", f"toto-{model}", "config.json")
    request = json.dumps({"context": context.tolist(), "horizon": horizon})

    result = subprocess.run(
        [RUST_BIN, "infer", "--gguf", gguf, "--config", config],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    quants = fc.get("quantiles", {})
    if "0.5" in quants:
        return np.array(quants["0.5"], dtype=np.float32)
    return np.array(fc["point"], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="22m",
                        choices=["4m", "22m", "313m", "1b", "2.5b"])
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8"])
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()

    context = make_context(128)
    print(f"Model: toto-{args.model}, dtype: {args.dtype}, horizon: {args.horizon}")

    print(f"Running Rust f32 reference …")
    f32_out = run_rust(args.model, "f32", context, args.horizon)
    print(f"  f32 shape: {f32_out.shape}, first 5: {f32_out[:5]}")

    if args.dtype == "f32":
        print(f"\nMax  |Δ|: 0.000000  (f32 vs f32)")
        print(f"Mean |Δ|: 0.000000")
        return

    print(f"Running Rust {args.dtype} …")
    rs_out = run_rust(args.model, args.dtype, context, args.horizon)

    diff = np.abs(f32_out - rs_out)
    print(f"\nComparison (f32 vs {args.dtype}) over {len(diff)} steps:")
    print(f"  Max  |Δ|: {diff.max():.6f}")
    print(f"  Mean |Δ|: {diff.mean():.6f}")


if __name__ == "__main__":
    main()
