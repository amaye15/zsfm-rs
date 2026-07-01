#!/usr/bin/env python3
"""Compare timesfm-rs Rust inference vs. Python TimesFM 2.5 200M.

Usage:
    python scripts/compare_python.py [--gguf gguf/timesfm-f32.gguf] [--horizon 64]
"""
import argparse
import json
import subprocess
import sys
import os
import numpy as np

# Add timesfm src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TIMESFM_SRC = os.path.join(REPO_ROOT, "..", "timesfm", "src")
if os.path.isdir(TIMESFM_SRC):
    sys.path.insert(0, TIMESFM_SRC)

QUANTILE_LABELS = ["point", "q0.10", "q0.20", "q0.30", "q0.40", "q0.50", "q0.60", "q0.70", "q0.80", "q0.90"]
QUANTILE_KEYS   = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]


def make_context(n=128):
    """Fixed deterministic time series: sine + trend."""
    t = np.linspace(0, 4 * np.pi, n)
    return (np.sin(t) + 0.05 * t + 1.0).astype(np.float32)


def run_rust(gguf: str, context: np.ndarray, horizon: int) -> np.ndarray:
    """Returns [horizon, 10] array: columns are point, q0.1..q0.9."""
    bin_path = os.path.join(REPO_ROOT, "target", "release", "timesfm-rs")
    request = json.dumps({"context": context.tolist(), "horizon": horizon})
    result = subprocess.run(
        [bin_path, "infer", "--gguf", gguf],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    point = np.array(fc["point"][:horizon], dtype=np.float32)
    quantiles = fc.get("quantiles", {})
    cols = [point]
    for k in QUANTILE_KEYS:
        cols.append(np.array(quantiles.get(k, [float("nan")] * horizon)[:horizon], dtype=np.float32))
    return np.stack(cols, axis=1)  # [horizon, 10]


def run_python(context: np.ndarray, horizon: int) -> np.ndarray:
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    import torch

    model = TimesFM_2p5_200M_torch_module()
    import huggingface_hub
    weights_path = huggingface_hub.hf_hub_download(
        "google/timesfm-2.5-200m-pytorch", "model.safetensors"
    )
    model.load_checkpoint(weights_path, torch_compile=False)

    results = model.forecast_naive(horizon, [context])
    # result: list of [horizon, 10] numpy arrays
    return results[0]  # [horizon, 10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="gguf/timesfm-f32.gguf")
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()

    context = make_context(128)
    print(f"Context length: {len(context)}, horizon: {args.horizon}")
    print(f"Context range: [{context.min():.3f}, {context.max():.3f}]")

    print("\nRunning Python TimesFM 2.5...")
    py_out = run_python(context, args.horizon)  # [horizon, 10]

    print(f"Running Rust inference ({args.gguf})...")
    rs_out = run_rust(args.gguf, context, args.horizon)  # [horizon, 10]

    py_out = py_out[:args.horizon]
    rs_out = rs_out[:args.horizon]
    assert py_out.shape == rs_out.shape, f"Shape mismatch: {py_out.shape} vs {rs_out.shape}"

    diff = np.abs(py_out - rs_out)
    print(f"\n{'':=<50}")
    print(f"{'Channel':<12}  {'MaxErr':>8}  {'MeanErr':>8}")
    print(f"{'':=<50}")
    for i, label in enumerate(QUANTILE_LABELS):
        print(f"{label:<12}  {diff[:, i].max():>8.4f}  {diff[:, i].mean():>8.4f}")
    print(f"{'':=<50}")
    print(f"{'ALL':12}  {diff.max():>8.4f}  {diff.mean():>8.4f}")

    print("\nFirst 8 point forecasts (Python vs Rust):")
    for t in range(min(8, args.horizon)):
        print(f"  t={t:2d}: py={py_out[t, 0]:.4f}  rs={rs_out[t, 0]:.4f}  diff={diff[t, 0]:.4f}")


if __name__ == "__main__":
    main()
