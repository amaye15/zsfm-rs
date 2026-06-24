#!/usr/bin/env python3
"""
Numerical comparison: Rust chronos-rs vs Python Chronos2Pipeline.

Both engines use the same local safetensors checkpoint (models/).
Input is a fixed 256-step sine+trend series; prediction_length=32.

Usage:
    python3 scripts/compare_python.py \
        --model-dir models \
        --gguf chronos-f16.gguf \
        --config models/config.json \
        --horizon 32
"""

import argparse
import json
import math
import subprocess
import sys

import numpy as np
import torch


def make_test_series(n: int = 256) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    return np.sin(2 * math.pi * t / 48) + 0.02 * t


def run_python(model_dir: str, series: np.ndarray, horizon: int) -> np.ndarray:
    from chronos.chronos2.pipeline import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(model_dir, torch_dtype=torch.float32)
    pipeline.model.eval()

    with torch.no_grad():
        preds = pipeline.predict(
            inputs=[torch.tensor(series, dtype=torch.float32)],
            prediction_length=horizon,
        )
    # preds[0]: (1, n_quantiles, prediction_length) → (n_quantiles, prediction_length)
    return preds[0].squeeze(0).numpy()


def run_rust(binary: str, gguf: str, config: str, series: np.ndarray,
             horizon: int) -> np.ndarray:
    request = json.dumps({"context": series.tolist(), "horizon": horizon})
    result = subprocess.run(
        [binary, "infer", "--gguf", gguf, "--config", config],
        input=request, capture_output=True, text=True, check=True,
    )
    fc = json.loads(result.stdout)["choices"][0]["forecast"]
    quants = fc.get("quantiles", {})
    keys = sorted(quants.keys(), key=float)
    rows = [np.array(quants[k], dtype=np.float32) for k in keys]
    if not rows:
        return np.array(fc["point"], dtype=np.float32)[None, :]
    return np.stack(rows)  # (n_quantiles, horizon)


def report(py_preds: np.ndarray, rs_preds: np.ndarray, quantile_labels: list[str]) -> None:
    diff = np.abs(py_preds - rs_preds)
    print(f"\n{'Quantile':<12} {'MaxAbsErr':>12} {'MeanAbsErr':>12} {'MedianAbsErr':>14}")
    print("-" * 54)
    for i, q in enumerate(quantile_labels):
        d = diff[i]
        print(f"{q:<12} {d.max():>12.6f} {d.mean():>12.6f} {np.median(d):>14.6f}")
    print("-" * 54)
    print(f"{'ALL':12} {diff.max():>12.6f} {diff.mean():>12.6f} {np.median(diff):>14.6f}")

    mid = len(quantile_labels) // 2
    q = quantile_labels[mid]
    py = py_preds[mid, :8]
    rs = rs_preds[mid, :8]
    print(f"\nSample [{q}] first 8 steps:")
    print(f"  Python: {' '.join(f'{v:.4f}' for v in py)}")
    print(f"    Rust: {' '.join(f'{v:.4f}' for v in rs)}")
    print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a,b in zip(py,rs))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--gguf", default="chronos-f16.gguf")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--binary", default="./target/release/chronos-rs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    quantiles: list[float] = cfg["chronos_config"]["quantiles"]
    quantile_labels = [f"q{q:.2f}" for q in quantiles]

    series = make_test_series(256)

    print(f"Input: {len(series)}-step sine+trend  |  Horizon: {args.horizon}")
    print(f"Quantiles: {quantile_labels}")

    print("\n--- Running Python (HuggingFace safetensors) ---")
    py_preds = run_python(args.model_dir, series, args.horizon)
    print(f"Python output shape: {py_preds.shape}")

    print("\n--- Running Rust (GGUF) ---")
    rs_preds = run_rust(args.binary, args.gguf, args.config, series, args.horizon)
    print(f"Rust output shape: {rs_preds.shape}")

    assert py_preds.shape == rs_preds.shape, (
        f"Shape mismatch: Python {py_preds.shape} vs Rust {rs_preds.shape}"
    )

    report(py_preds, rs_preds, quantile_labels)


if __name__ == "__main__":
    main()
