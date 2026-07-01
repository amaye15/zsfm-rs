#!/usr/bin/env python3
"""Compare tirex-rs inference output to the reference tirex-ts Python implementation."""

import argparse
import subprocess
import json
import math
import numpy as np


def run_rust(gguf_path: str, contexts: list[list[float]], horizon: int) -> list[dict]:
    """Run batch inference via stdin JSON. Returns list of forecast dicts."""
    request = json.dumps({"context": contexts, "horizon": horizon})
    result = subprocess.run(
        ["./target/release/tirex-rs", "infer", "--gguf", gguf_path],
        input=request, capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return [choice["forecast"] for choice in data["choices"]]


def run_python(context: list[float], horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Run TiRex Python reference and return (quantiles [T, Q], mean [T]), or None if unavailable."""
    try:
        import torch
        import tirex
    except ImportError:
        return None

    model = tirex.load_model("NX-AI/TiRex")
    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # [1, T]
    quantiles, mean = model._forecast_quantiles(ctx_tensor, prediction_length=horizon)
    # quantiles: [1, T, Q], mean: [1, T]
    return quantiles[0].numpy(), mean[0].numpy()


def mae(a, b):
    return float(np.mean(np.abs(np.array(a) - np.array(b))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="gguf/tirex-f32.gguf")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    t = np.arange(args.context_length)
    context = (np.sin(2 * math.pi * t / 24) + 0.1 * rng.standard_normal(args.context_length)).tolist()

    print(f"Context length: {args.context_length}, Horizon: {args.horizon}")
    print(f"GGUF: {args.gguf}")

    print("\nRunning Rust inference …")
    rust_forecasts = run_rust(args.gguf, [context], args.horizon)
    rust_fc = rust_forecasts[0]
    rust_point = rust_fc["point"]
    rust_q = rust_fc.get("quantiles", {})

    print("Running Python reference …")
    py_result = run_python(context, args.horizon)
    if py_result is None:
        print("Cannot compare: tirex-ts not installed (pip install tirex-ts)")
        print(f"Rust point[:5]: {[f'{v:.4f}' for v in rust_point[:5]]}")
        return

    py_q, py_mean = py_result
    # py_q: [T, Q] where Q=9 with quantiles 0.10..0.90
    py_point = py_mean.tolist()

    print(f"\nMax  |Δ|: {max(abs(a-b) for a,b in zip(rust_point, py_point)):.6f}")
    print(f"Mean |Δ|: {mae(rust_point, py_point):.6f}")
    print(f"Rust point[:5]:   {[f'{v:.4f}' for v in rust_point[:5]]}")
    print(f"Python point[:5]: {[f'{v:.4f}' for v in py_point[:5]]}")

    if rust_q:
        # Map "0.10", "0.20", ... to column index in py_q
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_key_to_idx = {f"{q:.2f}": i for i, q in enumerate(q_levels)}
        print("\nQuantile MAEs:")
        for qkey in sorted(rust_q.keys()):
            if qkey in q_key_to_idx:
                q_err = mae(rust_q[qkey], py_q[:, q_key_to_idx[qkey]])
                print(f"  {qkey}: {q_err:.6f}")


if __name__ == "__main__":
    main()
