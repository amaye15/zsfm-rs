#!/usr/bin/env python3
"""
Benchmark all Rust forecasting models on standard datasets,
including ensemble combinations.

All windows for a dataset are sent as a single batch call per model,
so the model binary is invoked once per model (not once per window).

Datasets
--------
  ETTh1, ETTh2    — hourly electricity transformer temperature (OT column)
  ETTm1, ETTm2    — 15-min electricity transformer temperature (OT column)
  electricity      — hourly electricity demand (single series)
  weather          — weather station (single series, Monash)
  weather_10min    — 10-min weather station, Germany 2020 (52k rows)
  jena             — 10-min Jena climate temperature 2009-2016 (420k rows)
  ili              — weekly CDC influenza-like illness (966 rows)
  exchange         — daily exchange rate USD→8 currencies (7.5k rows)
  traffic          — hourly SF Bay road occupancy (17.5k rows)

Evaluation
----------
  Rolling-window over the last TEST_ROWS rows of each dataset.
  Context length: 512 steps
  Horizon: 96 steps
  Windows: 30 evenly spaced
  Metric: MAE, RMSE, MASE (scaled by naive 1-step MAE on context)

  Ensembles evaluated for every non-trivial subset of selected models:
    mean      — simple average of predictions
    median    — per-timestep median
    weighted  — inverse-MAE weighted average (weights from individual results)
    trim      — trimmed mean: drop min+max prediction per timestep (≥3 models)
    softmax   — softmax of inverse-MAE weights (sharper than weighted)
    geometric — sign-preserving geometric mean
    online    — adaptive: uses each model's error on the previous window as weight
    adaptive  — Hedge algorithm: multiplicative weight update, optimal η=sqrt(2lnN/T) (ada)
    smooth    — EMA of inverse-MAE weights (blend of online and Hedge) (smo)
    select    — greedy model selection: pick single best by EMA-MAE (sel)
    per_horiz — per-horizon weights: different model mix per forecast step (wh)
    uncertain — IQR-weighted for Toto/Chronos, MAE-fallback for others (uq)

Usage
-----
  python benchmark/run_bench.py [--dataset ETTh1] [--windows 30]
  python benchmark/run_bench.py --ensemble-only   # skip solo results in table
"""

import argparse
import json
import math
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT  = Path(__file__).parent.parent
BENCH = Path(__file__).parent
DATA  = BENCH / "data"

MODELS = {
    "toto": {
        "bin":    ROOT / "toto-v2/toto-rs/target/release/toto-rs",
        "gguf":   ROOT / "toto-v2/toto-rs/gguf/toto-4m-f32.gguf",
        "config": ROOT / "toto-v2/toto-rs/models/toto-4m/config.json",
    },
    "chronos": {
        "bin":    ROOT / "chronos-v2/chronos-rs/target/release/chronos-rs",
        "gguf":   ROOT / "chronos-v2/chronos-rs/gguf/chronos-f32.gguf",
        "config": ROOT / "chronos-v2/chronos-rs/models/config.json",
    },
    "timesfm": {
        "bin":    ROOT / "timesfm-v2.5/timesfm-rs/target/release/timesfm-rs",
        "gguf":   ROOT / "timesfm-v2.5/timesfm-rs/gguf/timesfm-f32.gguf",
    },
    "sundial": {
        "bin":    ROOT / "timer-v3/sundial-rs/target/release/sundial-rs",
        "gguf":   ROOT / "timer-v3/sundial-rs/gguf/sundial-f32.gguf",
    },
    "ttm": {
        "bin":    ROOT / "ttm-v1/ttm-rs/target/release/ttm-rs",
        "gguf":   ROOT / "ttm-v1/ttm-rs/gguf/ttm-f32.gguf",
        "config": ROOT / "ttm-v1/ttm-rs/models/config.json",
    },
    "lag_llama": {
        "bin":  ROOT / "lag-llama-v1/lag-llama-rs/target/release/lag-llama-rs",
        "gguf": ROOT / "lag-llama-v1/lag-llama-rs/gguf/lag_llama-f32.gguf",
    },
    "moment": {
        "bin":  ROOT / "moment-v1/moment-rs/target/release/moment-rs",
        "gguf": ROOT / "moment-v1/moment-rs/gguf/moment-f32.gguf",
    },
    "moirai": {
        "bin":  ROOT / "moirai-v1/moirai-rs/target/release/moirai-rs",
        "gguf": ROOT / "moirai-v1/moirai-rs/gguf/moirai-f32.gguf",
    },
    "moirai2": {
        "bin":  ROOT / "moirai-v2/moirai-2-rs/target/release/moirai-2-rs",
        "gguf": ROOT / "moirai-v2/moirai-2-rs/gguf/moirai2-f32.gguf",
    },
    "flowstate": {
        "bin":    ROOT / "flowstate-r1/flowstate-r1-rs/target/release/flowstate-r1-rs",
        "gguf":   ROOT / "flowstate-r1/flowstate-r1-rs/gguf/flowstate-r1-f32.gguf",
        "config": ROOT / "flowstate-r1/flowstate-r1-rs/models/config.json",
    },
}

DATASETS = {
    # ETT — electricity transformer temperature (Informer paper)
    "ETTh1":           {"file": DATA / "ETTh1.csv",              "col": "OT",    "test_rows": 2880},
    "ETTh2":           {"file": DATA / "ETTh2.csv",              "col": "OT",    "test_rows": 2880},
    "ETTm1":           {"file": DATA / "ETTm1.csv",              "col": "OT",    "test_rows": 11520},
    "ETTm2":           {"file": DATA / "ETTm2.csv",              "col": "OT",    "test_rows": 11520},
    # Energy / demand
    "electricity":     {"file": DATA / "electricity_h1.csv",     "col": "value", "test_rows": 2880},
    "solar":           {"file": DATA / "solar_1h.csv",           "col": "value", "test_rows": 2000},
    "wind":            {"file": DATA / "wind_farms_h1.csv",      "col": "value", "test_rows": 2000},
    "aus_electricity": {"file": DATA / "aus_electricity_30min.csv","col":"value", "test_rows": 20000},
    # Weather / climate
    "weather":         {"file": DATA / "weather_s1.csv",         "col": "value", "test_rows": 1000},
    "weather_10min":   {"file": DATA / "weather_10min.csv",      "col": "OT",    "test_rows": 10560},
    "jena":            {"file": DATA / "jena_10min.csv",         "col": "value", "test_rows": 10560},
    "melbourne_temp":  {"file": DATA / "melbourne_temp.csv",     "col": "value", "test_rows": 365},
    "co2":             {"file": DATA / "co2_weekly.csv",         "col": "value", "test_rows": 300},
    # Astronomy / geophysics
    "sunspot_daily":   {"file": DATA / "sunspot_daily.csv",      "col": "value", "test_rows": 5000},
    "sunspot_monthly": {"file": DATA / "sunspot_monthly.csv",    "col": "value", "test_rows": 300},
    # Health — weekly CDC influenza-like illness
    "ili":             {"file": DATA / "ili.csv",                "col": "OT",    "test_rows": 200},
    # Finance — daily exchange rates, M4 competition
    "exchange":        {"file": DATA / "exchange_rate.csv",      "col": "OT",    "test_rows": 1500},
    "m4_daily":        {"file": DATA / "m4_daily.csv",           "col": "value", "test_rows": 2000},
    # Transport / urban
    "traffic":         {"file": DATA / "traffic_h1.csv",         "col": "OT",    "test_rows": 2880},
    "pedestrian":      {"file": DATA / "pedestrian_counts.csv",  "col": "value", "test_rows": 5000},
    # Hydrology
    "saugeeen":        {"file": DATA / "saugeeen_river.csv",     "col": "value", "test_rows": 2000},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_series(dataset_name: str) -> np.ndarray:
    import csv as _csv
    cfg = DATASETS[dataset_name]
    with open(cfg["file"]) as f:
        reader = _csv.DictReader(f)
        col = cfg["col"]
        return np.array([float(row[col]) for row in reader if row[col].strip()])


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def _cmd_for_model(model_name: str) -> list[str]:
    """Build the infer subcommand for a given model."""
    cfg = MODELS[model_name]
    cmd = [str(cfg["bin"]), "infer", "--gguf", str(cfg["gguf"])]
    if "config" in cfg:
        cmd += ["--config", str(cfg["config"])]
    return cmd


def collect_windows(
    model_name: str,
    series: np.ndarray,
    test_rows: int,
    context_len: int,
    horizon: int,
    n_windows: int,
) -> tuple[list, float, int, list]:
    """
    Run all rolling windows as a single batch inference call.

    Returns (windows, mean_lat_ms, error_count, unc_list).
      windows:  list length n_windows of (pred, truth, context) or None on failure.
      unc_list: per-window mean IQR for quantile models (nan for point models).
      mean_lat_ms: total elapsed / n_valid_windows * 1000  (per-window cost).
    """
    train_end = len(series) - test_rows
    max_start  = len(series) - horizon
    starts     = np.linspace(train_end, max_start - 1, n_windows, dtype=int)

    # Separate valid windows up front so we can build a clean batch
    valid = []   # (orig_idx, context_arr, truth_arr)
    for i, start in enumerate(starts):
        ctx_start = max(0, start - context_len)
        context   = series[ctx_start:start]
        truth     = series[start:start + horizon]
        if len(truth) >= horizon and len(context) >= 2:
            valid.append((i, context, truth))

    windows  = [None] * n_windows
    unc_list = [float("nan")] * n_windows

    if not valid:
        return windows, float("nan"), n_windows, unc_list

    request = json.dumps({
        "context": [ctx.tolist() for _, ctx, _ in valid],
        "horizon": horizon,
    })
    cmd = _cmd_for_model(model_name)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, input=request, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [{model_name}] ERROR — {e.stderr[-300:]}", file=sys.stderr)
        return windows, float("nan"), len(valid), unc_list
    elapsed = time.perf_counter() - t0

    try:
        data    = json.loads(proc.stdout.strip())
        choices = data["choices"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  [{model_name}] parse error — {e}", file=sys.stderr)
        return windows, float("nan"), len(valid), unc_list

    errors = 0
    for j, (orig_i, ctx, truth) in enumerate(valid):
        if j >= len(choices):
            errors += 1
            continue
        fc    = choices[j]["forecast"]
        point = np.array(fc["point"][:horizon], dtype=float)
        if len(point) < horizon:
            errors += 1
            continue
        quants = fc.get("quantiles", {})
        if "0.1" in quants and "0.9" in quants:
            unc = float(np.mean(
                np.array(quants["0.9"][:horizon]) - np.array(quants["0.1"][:horizon])
            ))
        else:
            unc = float("nan")
        windows[orig_i]  = (point, truth, ctx)
        unc_list[orig_i] = unc

    mean_lat_ms = elapsed / len(valid) * 1000
    return windows, mean_lat_ms, errors, unc_list


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mase(pred: np.ndarray, truth: np.ndarray, context: np.ndarray) -> float:
    naive_mae = np.mean(np.abs(np.diff(context)))
    if naive_mae == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - truth)) / naive_mae)


def metrics_from_windows(windows: list) -> dict:
    """windows: list of (pred, truth, context). Returns aggregated metrics."""
    maes, rmses, mases = [], [], []
    for pred, truth, ctx in windows:
        maes.append(float(np.mean(np.abs(pred - truth))))
        rmses.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
        m = mase(pred, truth, ctx)
        if not math.isnan(m):
            mases.append(m)
    if not maes:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MASE": float("nan")}
    return {
        "MAE":  float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "MASE": float(np.mean(mases)) if mases else float("nan"),
    }


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------

_SOFTMAX_TEMP = 0.5   # lower = sharper weight concentration on best model


def _combine(preds: np.ndarray, method: str, w: np.ndarray | None) -> np.ndarray:
    """
    preds: [n_models, horizon]
    Returns combined [horizon] prediction using the given method.
    """
    n = preds.shape[0]

    if method == "mean":
        return preds.mean(axis=0)

    elif method == "median":
        return np.median(preds, axis=0)

    elif method == "weighted":
        wn = w / w.sum()
        return (preds * wn[:, None]).sum(axis=0)

    elif method == "trim":
        if n <= 2:
            return preds.mean(axis=0)
        sorted_p = np.sort(preds, axis=0)
        return sorted_p[1:-1].mean(axis=0)

    elif method == "softmax":
        logits = w / w.sum() / _SOFTMAX_TEMP
        exp_w  = np.exp(logits - logits.max())
        wn     = exp_w / exp_w.sum()
        return (preds * wn[:, None]).sum(axis=0)

    elif method == "geometric":
        sign  = np.sign(np.median(preds, axis=0))
        sign[sign == 0] = 1.0
        log_abs = np.log(np.abs(preds) + 1e-9)
        geomean = np.exp(log_abs.mean(axis=0))
        return sign * geomean

    else:
        raise ValueError(f"Unknown method: {method}")


def ensemble_windows(
    all_windows: dict[str, list],
    model_names: list[str],
    method: str,
    static_weights: dict[str, float] | None = None,
    all_unc: dict[str, list] | None = None,
) -> list:
    """
    Build ensemble windows from per-model window lists.
    Only includes positions where ALL selected models have a valid prediction.
    Online methods update state after observing each window's truth (causal).
    """
    n        = len(next(iter(all_windows.values())))
    n_models = len(model_names)
    result   = []

    online_inv_mae = np.ones(n_models, dtype=float)

    hedge_w   = np.ones(n_models, dtype=float)
    HEDGE_ETA = np.sqrt(2.0 * np.log(max(n_models, 2)) / max(n, 1))

    smo_inv_mae = np.ones(n_models, dtype=float)
    SMO_ALPHA   = 0.5

    sel_cum_mae = np.ones(n_models, dtype=float)
    sel_n_valid = 0

    wh_inv_mae: np.ndarray | None = None

    uq_fallback_mae = np.ones(n_models, dtype=float)

    for i in range(n):
        entries = [all_windows[m][i] for m in model_names]
        if any(e is None for e in entries):
            continue
        preds   = np.stack([e[0] for e in entries])
        truth   = entries[0][1]
        context = entries[0][2]

        if method == "online":
            w        = online_inv_mae.copy()
            combined = _combine(preds, "weighted", w)
            per_mae  = np.mean(np.abs(preds - truth[None, :]), axis=1)
            online_inv_mae = 1.0 / np.clip(per_mae, 1e-12, None)

        elif method == "adaptive":
            w        = hedge_w / hedge_w.sum()
            combined = (preds * w[:, None]).sum(axis=0)
            losses   = np.abs(preds - truth[None, :]).mean(axis=1)
            hedge_w *= np.exp(-HEDGE_ETA * losses)
            hedge_w /= hedge_w.sum()

        elif method == "smooth":
            w        = smo_inv_mae / smo_inv_mae.sum()
            combined = (preds * w[:, None]).sum(axis=0)
            per_mae  = np.mean(np.abs(preds - truth[None, :]), axis=1)
            cur_inv  = 1.0 / np.clip(per_mae, 1e-12, None)
            smo_inv_mae = SMO_ALPHA * cur_inv + (1.0 - SMO_ALPHA) * smo_inv_mae

        elif method == "select":
            if sel_n_valid == 0:
                combined = preds.mean(axis=0)
            else:
                combined = preds[np.argmin(sel_cum_mae)]
            per_mae     = np.mean(np.abs(preds - truth[None, :]), axis=1)
            sel_cum_mae = (1.0 - SMO_ALPHA) * sel_cum_mae + SMO_ALPHA * per_mae
            sel_n_valid += 1

        elif method == "per_horiz":
            if wh_inv_mae is None:
                wh_inv_mae = np.ones((n_models, len(truth)), dtype=float)
                combined   = preds.mean(axis=0)
            else:
                w_h      = wh_inv_mae / wh_inv_mae.sum(axis=0, keepdims=True)
                combined = (preds * w_h).sum(axis=0)
            per_step_err = np.abs(preds - truth[None, :])
            step_inv     = 1.0 / np.clip(per_step_err, 1e-12, None)
            wh_inv_mae   = SMO_ALPHA * step_inv + (1.0 - SMO_ALPHA) * wh_inv_mae

        elif method == "uncertain":
            unc_vals = np.array([
                all_unc[m][i]
                if (all_unc and m in all_unc and not math.isnan(all_unc[m][i]))
                else uq_fallback_mae[j]
                for j, m in enumerate(model_names)
            ])
            w        = 1.0 / np.clip(unc_vals, 1e-12, None)
            combined = _combine(preds, "weighted", w)
            per_mae  = np.mean(np.abs(preds - truth[None, :]), axis=1)
            uq_fallback_mae = per_mae.copy()

        else:
            if static_weights is not None:
                w = np.array([static_weights[m] for m in model_names])
            else:
                w = np.ones(n_models)
            combined = _combine(preds, method, w)

        result.append((combined, truth, context))
    return result


METHODS = {
    "mean":      "μ",
    "median":    "~",
    "weighted":  "w",
    "trim":      "tr",
    "softmax":   "sfx",
    "geometric": "geo",
    "online":    "onl",
    "adaptive":  "ada",
    "smooth":    "smo",
    "select":    "sel",
    "per_horiz": "wh",
    "uncertain": "uq",
}

_WEIGHT_METHODS = {"weighted", "softmax"}


def label(models: tuple[str, ...], method: str) -> str:
    abbrev = {"toto": "T", "chronos": "C", "timesfm": "F", "sundial": "S",
              "ttm": "K", "lag_llama": "L", "moment": "M", "moirai": "O", "moirai2": "P",
              "flowstate": "W"}
    parts  = "".join(abbrev.get(m, m[0].upper()) for m in models)
    return f"{parts}({METHODS[method]})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset",  default="ETTh1", choices=list(DATASETS))
    parser.add_argument("--models",   nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--context",  type=int, default=512)
    parser.add_argument("--horizon",  type=int, default=96)
    parser.add_argument("--windows",  type=int, default=30)
    parser.add_argument("--ensemble-only", action="store_true",
                        help="Only show ensemble rows (still runs all models)")
    args = parser.parse_args()

    print(f"\nDataset : {args.dataset}")
    print(f"Context : {args.context}   Horizon : {args.horizon}   Windows : {args.windows}\n")

    cfg    = DATASETS[args.dataset]
    series = load_series(args.dataset)
    print(f"Series length: {len(series)}  |  Test rows: {cfg['test_rows']}\n")

    all_windows: dict[str, list]  = {}
    all_unc:     dict[str, list]  = {}
    solo_metrics: dict[str, dict] = {}
    solo_lat:    dict[str, float] = {}

    for model_name in args.models:
        print(f"Running {model_name} (batch) ...", flush=True)
        windows, lat_ms, errors, unc_list = collect_windows(
            model_name, series,
            cfg["test_rows"], args.context, args.horizon, args.windows,
        )
        all_windows[model_name] = windows
        all_unc[model_name]     = unc_list
        valid = [w for w in windows if w is not None]
        m = metrics_from_windows(valid)
        solo_metrics[model_name] = m
        solo_lat[model_name]     = lat_ms
        print(f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MASE={m['MASE']:.3f}  "
              f"lat={lat_ms:.0f}ms/window  errors={errors}")

    inv_mae = {m: 1.0 / solo_metrics[m]["MAE"] for m in args.models
               if not math.isnan(solo_metrics[m]["MAE"]) and solo_metrics[m]["MAE"] > 0}

    model_list       = args.models
    ensemble_results = {}

    for size in range(2, len(model_list) + 1):
        for combo in combinations(model_list, size):
            for method in METHODS:
                if method == "trim" and size < 3:
                    continue
                if method in _WEIGHT_METHODS and sum(
                    1 for m in combo if m in inv_mae and inv_mae[m] > 0
                ) < 2:
                    continue

                w  = ({m: inv_mae.get(m, 1e-9) for m in combo}
                      if method in _WEIGHT_METHODS else None)
                ew = ensemble_windows(all_windows, list(combo), method, w, all_unc=all_unc)
                if not ew:
                    continue
                ensemble_results[label(combo, method)] = metrics_from_windows(ew)

    col_w = 20
    print("\n" + "=" * 68)
    print(f"{'Model/Ensemble':<{col_w}} {'MAE':>8} {'RMSE':>8} {'MASE':>8}  {'vs best solo':>12}")
    print("-" * 68)

    best_solo_mae = min(solo_metrics[m]["MAE"] for m in args.models
                        if not math.isnan(solo_metrics[m]["MAE"]))

    def row(name, res, lat=None):
        delta = res["MAE"] - best_solo_mae
        sign  = "+" if delta >= 0 else ""
        lat_s = f"  {lat:.0f}ms" if lat is not None else ""
        print(f"{name:<{col_w}} {res['MAE']:>8.4f} {res['RMSE']:>8.4f} {res['MASE']:>8.3f}"
              f"  {sign}{delta:>+.4f}{lat_s}")

    if not args.ensemble_only:
        print("--- individual ---")
        for m in args.models:
            row(m, solo_metrics[m], solo_lat[m])
        print("--- ensembles ---")

    sorted_ens = sorted(ensemble_results.items(), key=lambda kv: kv[1]["MAE"])
    for name, res in sorted_ens:
        row(name, res)

    print("=" * 68)
    print("\nEnsemble key:  T=toto  C=chronos  F=timesfm  S=sundial  K=ttm  "
          "L=lag_llama  M=moment  O=moirai  P=moirai2  W=flowstate")
    print("  (μ)=mean  (~)=median  (w)=inv-MAE weighted  (tr)=trimmed mean")
    print("  (sfx)=softmax  (geo)=geometric mean  (onl)=online  (ada)=Hedge  "
          "(smo)=smooth  (sel)=best-model  (wh)=per-horizon  (uq)=uncertainty")


if __name__ == "__main__":
    main()
