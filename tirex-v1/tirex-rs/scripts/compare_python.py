#!/usr/bin/env python3
"""Compare tirex-rs inference output to the reference tirex-ts Python implementation."""

import argparse
import subprocess
import json
import math
import numpy as np

def run_rust(gguf_path: str, context: list[float], horizon: int) -> dict:
    ctx_str = ",".join(str(v) for v in context)
    result = subprocess.run(
        ["./target/release/tirex-rs", "infer",
         "--gguf", gguf_path,
         "--data", ctx_str,
         "--horizon", str(horizon),
         "--all-outputs"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)

def run_python(context: list[float], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Run TiRex Python reference and return (quantiles [T, Q], mean [T])."""
    try:
        import torch
        import tirex
    except ImportError:
        raise SystemExit("Install tirex-ts: pip install tirex-ts")

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
    # Synthetic sine wave context
    t = np.arange(args.context_length)
    context = (np.sin(2 * math.pi * t / 24) + 0.1 * rng.standard_normal(args.context_length)).tolist()

    print(f"Context length: {args.context_length}, Horizon: {args.horizon}")
    print(f"GGUF: {args.gguf}")

    print("\nRunning Rust inference …")
    rust_out = run_rust(args.gguf, context, args.horizon)
    rust_point = rust_out["choices"][0]["forecast"]["point"]
    rust_q = rust_out["choices"][0]["forecast"].get("quantiles", {})

    print("Running Python reference …")
    py_q, py_mean = run_python(context, args.horizon)
    # py_q: [T, Q] where Q=9 with quantiles 0.1..0.9
    py_point = py_mean.tolist()

    print(f"\nPoint forecast MAE (Rust vs Python): {mae(rust_point, py_point):.6f}")
    print(f"Rust point[:5]:   {[f'{v:.4f}' for v in rust_point[:5]]}")
    print(f"Python point[:5]: {[f'{v:.4f}' for v in py_point[:5]]}")

    if rust_q:
        q_names = sorted(rust_q.keys())
        q_idx_map = {f"q{q:.1f}": i for i, q in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])}
        print("\nQuantile MAEs:")
        for qname in q_names:
            if qname in q_idx_map:
                q_err = mae(rust_q[qname], py_q[:, q_idx_map[qname]])
                print(f"  {qname}: {q_err:.6f}")

if __name__ == "__main__":
    main()
