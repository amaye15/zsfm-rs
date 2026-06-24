#!/usr/bin/env python3
"""
Compare Python reference vs Rust GGUF inference for fp32, fp16, and int8.

Runs each model's compare_python.py script across all three dtypes and
reports max and mean absolute error in a single summary table.

Usage:
    python benchmark/compare_precision.py [--horizon N]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    {
        "name":     "moirai-2",
        "dir":      "moirai-v2/moirai-2-rs",
        "project":  "moirai-v2",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/moirai2-f32.gguf", "f16": "gguf/moirai2-f16.gguf", "q8": "gguf/moirai2-q8.gguf"},
        "extra":    ["--horizon", "{horizon}"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "moirai",
        "dir":      "moirai-v1/moirai-rs",
        "project":  "moirai-v1",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/moirai-f32.gguf", "f16": "gguf/moirai-f16.gguf", "q8": "gguf/moirai-q8.gguf"},
        "extra":    ["--horizon", "{horizon}"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "lag-llama",
        "dir":      "lag-llama-v1/lag-llama-rs",
        "project":  "lag-llama-v1",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/lag_llama-f32.gguf", "f16": "gguf/lag_llama-f16.gguf", "q8": "gguf/lag_llama-q8.gguf"},
        "extra":    ["--horizon", "{horizon}"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "moment",
        "dir":      "moment-v1/moment-rs",
        "project":  "moment-v1",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/moment-f32.gguf", "f16": "gguf/moment-f16.gguf", "q8": "gguf/moment-q8.gguf"},
        "extra":    ["--horizon", "{horizon}"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "sundial",
        "dir":      "timer-v3/sundial-rs",
        "project":  "timer-v3",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/sundial-f32.gguf", "f16": "gguf/sundial-f16.gguf", "q8": "gguf/sundial-q8.gguf"},
        "extra":    [],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "chronos",
        "dir":      "chronos-v2/chronos-rs",
        "project":  "chronos-v2",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/chronos-f32.gguf", "f16": "gguf/chronos-f16.gguf", "q8": "gguf/chronos-q8.gguf"},
        "extra":    ["--horizon", "32"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "timesfm",
        "dir":      "timesfm-v2.5/timesfm-rs",
        "project":  "timesfm-v2.5",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/timesfm-f32.gguf", "f16": "gguf/timesfm-f16.gguf", "q8": "gguf/timesfm-q8.gguf"},
        "extra":    ["--horizon", "{horizon}"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "ttm",
        "dir":      "ttm-v1/ttm-rs",
        "project":  "ttm-v1",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/ttm-f32.gguf", "f16": "gguf/ttm-f16.gguf", "q8": "gguf/ttm-q8.gguf"},
        "extra":    ["--horizon", "96"],
        "gguf_arg": "--gguf",
    },
    {
        "name":     "toto",
        "dir":      "toto-v2/toto-rs",
        "project":  "toto-v2",
        "script":   "scripts/compare_python.py",
        "ggufs":    {"f32": "gguf/toto-22m-f32.gguf", "f16": "gguf/toto-22m-f16.gguf", "q8": "gguf/toto-22m-q8.gguf"},
        "extra":    ["--model", "22m", "--dtype", "{dtype}"],
        "gguf_arg": None,  # toto script ignores --gguf, uses --model/--dtype
    },
]


def run_comparison(model_dir: str, project_dir: str, script: str, gguf_arg: str | None,
                   gguf_path: str, extra_args: list[str]) -> tuple[float, float] | None:
    cmd = ["uv", "run", "--project", project_dir, "python", script]
    if gguf_arg is not None:
        cmd += [gguf_arg, gguf_path]
    cmd += extra_args
    try:
        result = subprocess.run(
            cmd,
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()[-300:]}", file=sys.stderr)
        return None

    output = result.stdout + result.stderr
    # Parse "Max  |Δ|: X.XXXXXX" and "Mean |Δ|: X.XXXXXX"
    max_match  = re.search(r"Max\s+\|[Δd]\|:\s*([\d.]+)", output)
    mean_match = re.search(r"Mean\s+\|[Δd]\|:\s*([\d.]+)", output)
    # Fallback: timesfm-style "ALL   max   mean" row
    if not max_match:
        all_match = re.search(r"ALL\s+([\d.]+)\s+([\d.]+)", output)
        if all_match:
            return float(all_match.group(1)), float(all_match.group(2))
    if max_match and mean_match:
        return float(max_match.group(1)), float(mean_match.group(1))
    # Rust ran but no Python reference available
    if "Cannot compare" in output or "skipping" in output.lower():
        return (-1.0, -1.0)  # sentinel: Rust OK, no Python ref
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=64,
                        help="Forecast horizon (default: 64)")
    args = parser.parse_args()

    horizon = args.horizon
    dtypes  = ["f32", "f16", "q8"]

    print(f"\nPython vs Rust precision comparison  (horizon={horizon})")
    print("=" * 72)
    print(f"{'Model':<14}  {'dtype':<5}  {'Max |Δ|':>10}  {'Mean |Δ|':>10}  {'vs f32 max':>12}")
    print("-" * 72)

    for model in MODELS:
        model_dir   = os.path.join(ROOT, model["dir"])
        project_dir = os.path.join(ROOT, model["project"])
        script      = model["script"]
        base_max    = None

        for dtype in dtypes:
            gguf = model["ggufs"][dtype]
            extra = [a.replace("{horizon}", str(horizon)).replace("{dtype}", dtype)
                     for a in model["extra"]]
            sys.stdout.write(f"  {model['name']:<12}  {dtype:<5}  running…\r")
            sys.stdout.flush()

            result = run_comparison(model_dir, project_dir, script, model["gguf_arg"], gguf, extra)

            if result is None:
                print(f"  {model['name']:<12}  {dtype:<5}  {'FAILED':>10}  {'FAILED':>10}  {'':>12}")
                continue

            max_err, mean_err = result
            if max_err == -1.0 and mean_err == -1.0:
                print(f"  {model['name']:<12}  {dtype:<5}  {'N/A':>10}  {'N/A':>10}  {'(no py ref)':>12}")
                continue

            if dtype == "f32":
                base_max = max_err
                vs_f32 = "—"
            else:
                vs_f32 = f"+{max_err - base_max:.6f}" if base_max is not None else "N/A"

            print(f"  {model['name']:<12}  {dtype:<5}  {max_err:>10.6f}  {mean_err:>10.6f}  {vs_f32:>12}")

        print()

    print("-" * 72)


if __name__ == "__main__":
    main()
