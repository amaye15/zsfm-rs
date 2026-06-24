#!/usr/bin/env python3
"""
Benchmark all Toto GGUF files: file size, inference speed, and quantization error.

Usage:
    python benchmark.py                  # all models & dtypes
    python benchmark.py --model toto-4m  # one model only
    python benchmark.py --ctx 256        # custom context length
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

GGUF_DIR = "./gguf"
MODELS_DIR = "./models"
BIN = "./target/release/toto-rs"

MODELS = [
    ("toto-4m",   "Datadog/Toto-2.0-4m"),
    ("toto-22m",  "Datadog/Toto-2.0-22m"),
    ("toto-313m", "Datadog/Toto-2.0-313m"),
    ("toto-1b",   "Datadog/Toto-2.0-1B"),
    ("toto-2.5b", "Datadog/Toto-2.0-2.5B"),
]
DTYPES = ["f32", "f16", "q8"]

PRED_LEN = 64
N_TIMED_RUNS = 3


def make_context(ctx_len: int) -> list[float]:
    return [100.0 * math.sin(2 * math.pi * t / 32) + 500.0 for t in range(ctx_len)]


def gguf_path(tag: str, dtype: str) -> str:
    return os.path.join(GGUF_DIR, f"{tag}-{dtype}.gguf")


def config_path(tag: str) -> str:
    return os.path.join(MODELS_DIR, tag, "config.json")


def run_inference(gguf: str, config: str, context: list[float]) -> np.ndarray | None:
    """Run toto-rs infer and return quantile array [9][pred_len], or None on error."""
    request = json.dumps({"context": context, "horizon": PRED_LEN})
    try:
        result = subprocess.run(
            [BIN, "infer", "--gguf", gguf, "--config", config],
            input=request,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e.stderr.strip().splitlines()[-1] if e.stderr else e}", file=sys.stderr)
        return None

    data = json.loads(result.stdout)
    fc = data["choices"][0]["forecast"]
    quants = fc.get("quantiles", {})
    keys = sorted(quants.keys(), key=float)
    if not keys:
        return None
    return np.array([quants[k] for k in keys], dtype=np.float32)  # [9, pred_len]


def time_inference(gguf: str, config: str, context: list[float], n_runs: int) -> float | None:
    """Return mean wall-clock seconds for n_runs inference calls."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ok = run_inference(gguf, config, context)
        if ok is None:
            return None
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def file_size_gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restrict to one tag, e.g. toto-4m")
    parser.add_argument("--ctx", type=int, default=512, help="context length (default 512)")
    args = parser.parse_args()

    ctx_len = args.ctx
    context = make_context(ctx_len)

    models = [(t, hf) for t, hf in MODELS if args.model is None or t == args.model]
    if not models:
        sys.exit(f"No model matching --model {args.model!r}")

    w_tag = 10
    w_dtype = 6
    w_size = 8
    w_time = 9
    w_mae = 10

    header = (
        f"{'Model':<{w_tag}} {'dtype':<{w_dtype}} {'Size(GB)':>{w_size}} "
        f"{'Time(s)':>{w_time}} {'MAE_vs_F32':>{w_mae}}"
    )
    sep = "-" * len(header)
    print(f"\nContext={ctx_len} steps, prediction={PRED_LEN} steps, {N_TIMED_RUNS} timed runs\n")
    print(header)
    print(sep)

    for tag, _hf in models:
        cfg = config_path(tag)
        if not os.path.exists(cfg):
            print(f"{tag:<{w_tag}} [no config.json — run convert first]")
            continue

        f32_gguf = gguf_path(tag, "f32")
        f32_ref: np.ndarray | None = None
        if os.path.exists(f32_gguf):
            f32_ref = run_inference(f32_gguf, cfg, context)

        first_row = True
        for dtype in DTYPES:
            path = gguf_path(tag, dtype)
            model_col = tag if first_row else ""
            first_row = False

            if not os.path.exists(path):
                print(
                    f"{model_col:<{w_tag}} {dtype:<{w_dtype}} {'missing':>{w_size}} "
                    f"{'—':>{w_time}} {'—':>{w_mae}}"
                )
                continue

            size = file_size_gb(path)
            elapsed = time_inference(path, cfg, context, N_TIMED_RUNS)

            mae_str = "—"
            if f32_ref is not None and dtype != "f32":
                preds = run_inference(path, cfg, context)
                if preds is not None:
                    mae = float(np.abs(preds - f32_ref).mean())
                    mae_str = f"{mae:.5f}"

            elapsed_str = f"{elapsed:.2f}" if elapsed is not None else "—"
            print(
                f"{model_col:<{w_tag}} {dtype:<{w_dtype}} {size:>{w_size}.2f} "
                f"{elapsed_str:>{w_time}} {mae_str:>{w_mae}}"
            )

        if not first_row:
            print(sep)


if __name__ == "__main__":
    main()
