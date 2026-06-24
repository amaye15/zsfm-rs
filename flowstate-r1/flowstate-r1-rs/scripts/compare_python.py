#!/usr/bin/env python3
"""
Numerical comparison: Rust flowstate-r1-rs vs Python FlowStateForPrediction.

Both engines use the same local safetensors checkpoint (models/).
Input is a fixed 256-step sine+trend series; prediction_length=24.

Usage:
    python3 scripts/compare_python.py \
        --model-dir models \
        --gguf gguf/flowstate-r1-f32.gguf \
        --config models/config.json \
        --horizon 24
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np


def make_test_series(n: int = 256) -> np.ndarray:
    t = np.arange(n, dtype=np.float32)
    return np.sin(2 * math.pi * t / 48) + 0.02 * t


def run_python(model_dir: str, series: np.ndarray, horizon: int) -> np.ndarray:
    try:
        import torch
        from tsfm_public import FlowStateForPrediction
    except ImportError:
        print("tsfm_public not installed. Install with:")
        print("  pip install git+https://github.com/ibm-granite/granite-tsfm.git")
        sys.exit(1)

    predictor = FlowStateForPrediction.from_pretrained(model_dir, torch_dtype=torch.float32)
    predictor.model.eval()

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    decoder_patch_len = cfg["decoder_patch_len"]
    scale_factor = decoder_patch_len / horizon

    with torch.no_grad():
        ts = torch.tensor(series, dtype=torch.float32).unsqueeze(-1).unsqueeze(1)
        preds = predictor(
            ts,
            scale_factor=scale_factor,
            prediction_length=horizon,
            batch_first=False,
        )

    if hasattr(preds, "quantile_outputs") and preds.quantile_outputs is not None:
        arr = preds.quantile_outputs[0, :, :, 0].detach().numpy()  # [n_q, pred_len]
    elif hasattr(preds, "last_hidden_state"):
        arr = preds.last_hidden_state.squeeze().detach().numpy()
    else:
        arr = preds.squeeze().detach().numpy()

    return arr.astype(np.float32)


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


def report(py_preds: np.ndarray, rs_preds: np.ndarray, quantile_labels) -> None:
    diff = np.abs(py_preds - rs_preds)
    print(f"\n{'Quantile':<12} {'MaxAbsErr':>12} {'MeanAbsErr':>12} {'MedianAbsErr':>14}")
    print("-" * 54)
    for i, q in enumerate(quantile_labels):
        d = diff[i]
        print(f"{q:<12} {d.max():>12.6f} {d.mean():>12.6f} {np.median(d):>14.6f}")
    print("-" * 54)
    print(f"{'ALL':12} {diff.max():>12.6f} {diff.mean():>12.6f} {np.median(diff):>14.6f}")

    for idx in [0, len(quantile_labels) // 2, len(quantile_labels) - 1]:
        q = quantile_labels[idx]
        py = py_preds[idx, :8]
        rs = rs_preds[idx, :8]
        print(f"\nSample [{q}] first 8 steps:")
        print(f"  Python: {' '.join(f'{v:.4f}' for v in py)}")
        print(f"    Rust: {' '.join(f'{v:.4f}' for v in rs)}")
        print(f"    Diff: {' '.join(f'{abs(a-b):.4f}' for a, b in zip(py, rs))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--gguf", default="gguf/flowstate-r1-f32.gguf")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--binary", default="./target/release/flowstate-r1-rs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    quantiles = cfg["quantiles"]
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

    if py_preds.shape != rs_preds.shape:
        print(f"Shape mismatch: Python {py_preds.shape} vs Rust {rs_preds.shape}")
        sys.exit(1)

    report(py_preds, rs_preds, quantile_labels)


if __name__ == "__main__":
    main()
