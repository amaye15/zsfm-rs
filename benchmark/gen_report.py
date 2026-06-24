#!/usr/bin/env python3
"""
Run the full benchmark across all datasets, all models, multiple horizons
and context lengths, then write benchmark.md.

Each run_bench.py call sends all rolling windows as a single batch, so the
model binary is loaded once per model per dataset rather than once per window.

Usage:
  python benchmark/gen_report.py [--main-windows 30] [--sweep-windows 15]
  python benchmark/gen_report.py --parallel 4   # run datasets concurrently
"""

import argparse
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT  = Path(__file__).parent.parent
BENCH = Path(__file__).parent
CACHE = BENCH / ".bench_cache.json"

_cache_lock = threading.Lock()

DATASETS_ORDER = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "electricity", "solar", "wind", "aus_electricity",
    "weather", "weather_10min", "jena", "melbourne_temp", "co2",
    "sunspot_daily", "sunspot_monthly",
    "ili",
    "exchange", "m4_daily",
    "traffic", "pedestrian",
    "saugeeen",
]

DATASET_META = {
    "ETTh1":          {"freq": "1h",    "domain": "Energy"},
    "ETTh2":          {"freq": "1h",    "domain": "Energy"},
    "ETTm1":          {"freq": "15min", "domain": "Energy"},
    "ETTm2":          {"freq": "15min", "domain": "Energy"},
    "electricity":    {"freq": "1h",    "domain": "Energy"},
    "solar":          {"freq": "1h",    "domain": "Energy"},
    "wind":           {"freq": "1h",    "domain": "Energy"},
    "aus_electricity":{"freq": "30min", "domain": "Energy"},
    "weather":        {"freq": "1h",    "domain": "Climate"},
    "weather_10min":  {"freq": "10min", "domain": "Climate"},
    "jena":           {"freq": "10min", "domain": "Climate"},
    "melbourne_temp": {"freq": "1d",    "domain": "Climate"},
    "co2":            {"freq": "1w",    "domain": "Climate"},
    "sunspot_daily":  {"freq": "1d",    "domain": "Astronomy"},
    "sunspot_monthly":{"freq": "1mo",   "domain": "Astronomy"},
    "ili":            {"freq": "1w",    "domain": "Health"},
    "exchange":       {"freq": "1d",    "domain": "Finance"},
    "m4_daily":       {"freq": "1d",    "domain": "Finance"},
    "traffic":        {"freq": "1h",    "domain": "Transport"},
    "pedestrian":     {"freq": "1h",    "domain": "Transport"},
    "saugeeen":       {"freq": "1d",    "domain": "Hydrology"},
}

SOLO_MODELS = ["toto", "chronos", "timesfm", "sundial", "ttm", "lag_llama", "moment", "moirai", "moirai2", "flowstate"]
MODEL_LABELS = {
    "toto": "Toto", "chronos": "Chronos",
    "timesfm": "TimesFM", "sundial": "Sundial",
    "ttm": "TTM", "lag_llama": "Lag-Llama",
    "moment": "Moment", "moirai": "Moirai",
    "moirai2": "Moirai-2", "flowstate": "FlowState",
}

HORIZONS      = [96, 192, 336, 720]
CONTEXTS      = [96, 256, 512, 1024]
MAIN_CONTEXT  = 512
MAIN_HORIZON  = 96
TOP_N_ENSEMBLES = 10


# ---------------------------------------------------------------------------
# Cache helpers (thread-safe)
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    with _cache_lock:
        if CACHE.exists():
            return json.loads(CACHE.read_text())
        return {}


def save_cache(cache: dict) -> None:
    with _cache_lock:
        CACHE.write_text(json.dumps(cache, indent=2))


def cache_key(dataset: str, context: int, horizon: int, windows: int) -> str:
    return f"{dataset}:ctx{context}:h{horizon}:w{windows}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, context: int, horizon: int, windows: int,
                cache: dict) -> str:
    key = cache_key(dataset, context, horizon, windows)
    with _cache_lock:
        if key in cache:
            return cache[key]

    cmd = [
        sys.executable, str(BENCH / "run_bench.py"),
        "--dataset", dataset,
        "--context", str(context),
        "--horizon", str(horizon),
        "--windows", str(windows),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"  ERROR {dataset} ctx={context} h={horizon}: {result.stderr[-300:]}",
              file=sys.stderr)
    out = result.stdout

    with _cache_lock:
        cache[key] = out
        CACHE.write_text(json.dumps(cache, indent=2))
    return out


def run_phase(
    configs: list[tuple],   # (dataset, context, horizon, windows)
    cache: dict,
    workers: int,
    label_fn,               # (dataset, context, horizon) -> str for progress
) -> dict:
    """
    Run a list of (dataset, context, horizon, windows) configs in parallel.
    Returns {(dataset, context, horizon): parsed_result}.
    """
    results = {}
    lock    = threading.Lock()

    def _run(cfg):
        ds, ctx, h, w = cfg
        raw    = run_dataset(ds, ctx, h, w, cache)
        parsed = parse_output(raw)
        print(f"  {label_fn(ds, ctx, h)}  "
              f"({sum(1 for m in SOLO_MODELS if m in parsed['solo'])} models)")
        with lock:
            results[(ds, ctx, h)] = parsed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_run, cfg) for cfg in configs]
        for f in as_completed(futs):
            exc = f.exception()
            if exc:
                print(f"  phase worker error: {exc}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(text: str) -> dict:
    result = {"solo": {}, "ensembles": [], "series_len": 0, "test_rows": 0}

    m = re.search(r"Series length:\s*(\d+)\s*\|.*?Test rows:\s*(\d+)", text)
    if m:
        result["series_len"] = int(m.group(1))
        result["test_rows"]  = int(m.group(2))

    table_m = re.search(r"={40,}.*?Model/Ensemble(.*?)={40,}", text, re.DOTALL)
    if not table_m:
        return result
    table = table_m.group(1)

    parts   = re.split(r"---\s*(individual|ensembles)\s*---", table)
    section = None
    for part in parts:
        part = part.strip()
        if part == "individual":
            section = "solo"
            continue
        elif part == "ensembles":
            section = "ensemble"
            continue
        if section is None or not part:
            continue

        for line in part.splitlines():
            line = line.strip()
            if not line or line.startswith("=") or line.startswith("-"):
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            name = tokens[0]
            try:
                mae   = float(tokens[1])
                rmse  = float(tokens[2])
                mase  = float(tokens[3])
                delta = float(tokens[4].lstrip("+")) if len(tokens) > 4 else 0.0
            except ValueError:
                continue

            if section == "solo":
                lat = None
                for t in tokens:
                    if t.endswith("ms"):
                        try: lat = float(t[:-2])
                        except ValueError: pass
                result["solo"][name] = {
                    "MAE": mae, "RMSE": rmse, "MASE": mase,
                    "lat_ms": lat, "delta": delta,
                }
            else:
                result["ensembles"].append({
                    "name": name, "MAE": mae, "RMSE": rmse,
                    "MASE": mase, "delta": delta,
                })

    result["ensembles"].sort(key=lambda x: x["MAE"])
    return result


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return f"{v:.{decimals}f}"


def md_table(headers, rows, align=None):
    if align is None:
        align = ["left"] + ["right"] * (len(headers) - 1)
    sep = [":---" if a == "left" else "---:" for a in align]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def bold_min(values: list[str], raw: list[float]) -> list[str]:
    try:
        min_val = min(v for v in raw if v == v)
    except ValueError:
        return values
    return [f"**{v}**" if abs(r - min_val) < 1e-9 else v
            for v, r in zip(values, raw)]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_sweep_table(
    sweep_results: dict,
    sweep_values: list,
    col_label: str,
    metric: str,
    model: str,
    decimals: int = 4,
) -> str:
    headers = ["Dataset"] + [f"{col_label}={v}" for v in sweep_values] + ["Trend"]
    rows = []
    for ds in DATASETS_ORDER:
        raw = []
        for v in sweep_values:
            solo = sweep_results.get((ds, MAIN_CONTEXT if col_label == "h" else v,
                                      v if col_label == "h" else MAIN_HORIZON), {}).get("solo", {})
            r = solo.get(model, {}).get(metric)
            raw.append(r if r is not None else float("nan"))
        cells = bold_min([fmt(r, decimals) for r in raw], raw)

        valid = [(v, r) for v, r in zip(sweep_values, raw) if r == r]
        if len(valid) >= 2:
            first, last = valid[0][1], valid[-1][1]
            pct   = (last - first) / abs(first) * 100 if first else 0
            trend = f"↑ {pct:+.0f}%" if pct > 0 else f"↓ {abs(pct):.0f}%"
        else:
            trend = "—"

        rows.append([ds] + cells + [trend])
    return md_table(headers, rows)


def build_avg_sweep_table(
    sweep_results: dict,
    sweep_values: list,
    col_label: str,
    metric: str = "MASE",
    decimals: int = 3,
) -> str:
    headers = ["Model"] + [f"{col_label}={v}" for v in sweep_values]
    rows = []
    for model in SOLO_MODELS:
        raw = []
        for v in sweep_values:
            vals = []
            for ds in DATASETS_ORDER:
                key = (ds, MAIN_CONTEXT if col_label == "h" else v,
                       v if col_label == "h" else MAIN_HORIZON)
                r = sweep_results.get(key, {}).get("solo", {}).get(model, {}).get(metric)
                if r is not None and r == r:
                    vals.append(r)
            raw.append(sum(vals) / len(vals) if vals else float("nan"))
        cells = bold_min([fmt(r, decimals) for r in raw], raw)
        rows.append([MODEL_LABELS[model]] + cells)
    return md_table(headers, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-windows",  type=int, default=30)
    parser.add_argument("--sweep-windows", type=int, default=15)
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of datasets to run concurrently (default: 1)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached results and re-run everything")
    args = parser.parse_args()

    cache = {} if args.no_cache else load_cache()

    # -----------------------------------------------------------------------
    # Phase 1: Main benchmark — context=512, horizon=96
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  PHASE 1: Main benchmark (ctx={MAIN_CONTEXT}, h={MAIN_HORIZON})"
          f"  [parallel={args.parallel}]")
    print("="*60)

    phase1_cfgs = [
        (ds, MAIN_CONTEXT, MAIN_HORIZON, args.main_windows)
        for ds in DATASETS_ORDER
    ]
    phase1_raw = run_phase(
        phase1_cfgs, cache, args.parallel,
        lambda ds, ctx, h: f"[{ds}]",
    )
    main_results: dict[str, dict] = {
        ds: phase1_raw[(ds, MAIN_CONTEXT, MAIN_HORIZON)]
        for ds in DATASETS_ORDER
        if (ds, MAIN_CONTEXT, MAIN_HORIZON) in phase1_raw
    }
    for ds in DATASETS_ORDER:
        solo = main_results.get(ds, {}).get("solo", {})
        for m in SOLO_MODELS:
            if m in solo:
                print(f"    {ds}/{m}: MAE={solo[m]['MAE']:.4f}")

    # -----------------------------------------------------------------------
    # Phase 2: Horizon sweep — context=512, horizons=[96,192,336,720]
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  PHASE 2: Horizon sweep (ctx={MAIN_CONTEXT}, h={HORIZONS})"
          f"  [parallel={args.parallel}]")
    print("="*60)

    phase2_cfgs = [
        (ds, MAIN_CONTEXT, h, args.sweep_windows)
        for ds in DATASETS_ORDER
        for h in HORIZONS
    ]
    horizon_results = run_phase(
        phase2_cfgs, cache, args.parallel,
        lambda ds, ctx, h: f"[{ds}] h={h}",
    )

    # -----------------------------------------------------------------------
    # Phase 3: Context sweep — horizon=96, contexts=[96,256,512,1024]
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  PHASE 3: Context sweep (h={MAIN_HORIZON}, ctx={CONTEXTS})"
          f"  [parallel={args.parallel}]")
    print("="*60)

    phase3_cfgs = [
        (ds, ctx, MAIN_HORIZON, args.sweep_windows)
        for ds in DATASETS_ORDER
        for ctx in CONTEXTS
    ]
    context_results = run_phase(
        phase3_cfgs, cache, args.parallel,
        lambda ds, ctx, h: f"[{ds}] ctx={ctx}",
    )

    # -----------------------------------------------------------------------
    # Write benchmark.md
    # -----------------------------------------------------------------------
    lines = []

    lines += [
        "# Time Series Forecasting Benchmark",
        "",
        f"**Date:** {date.today()}  ",
        f"**Main evaluation:** context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, "
        f"{args.main_windows} rolling windows  ",
        f"**Sweep evaluations:** {args.sweep_windows} rolling windows  ",
        "**Metric:** MAE (lower is better); MASE for cross-dataset comparison (scale-free)",
        "",
        "**Models:**",
        "| Model | Params | Architecture |",
        "|-------|--------|--------------|",
        "| Toto (4M) | 4M | Quantile transformer (Datadog) |",
        "| Chronos | ~200M | Tokenised probabilistic transformer (Amazon) |",
        "| TimesFM | ~200M | Patch-based decoder (Google) |",
        "| Sundial | 128M | Flow-matching decoder transformer (thuml) |",
        "| TTM | ~1M | Tiny time-mixer (IBM) |",
        "| Lag-Llama | ~24M | Lag-feature LLM decoder (ServiceNow) |",
        "| Moment | ~385M | Masked patch encoder, T5 attention (CMU) |",
        "| Moirai | ~311M | Universal forecasting encoder (Salesforce) |",
        "| Moirai-2 | ~311M | Universal forecasting decoder (Salesforce) |",
        "| FlowState | ~50M | SSM encoder-decoder with Legendre quantile basis (IBM) |",
        "",
        "**Ensemble strategies (all non-trivial subsets, size ≥ 2):**",
        "| Symbol | Strategy |",
        "|--------|----------|",
        "| `(μ)` | Simple mean |",
        "| `(~)` | Per-timestep median |",
        "| `(w)` | Inverse-MAE weighted |",
        "| `(tr)` | Trimmed mean (drop min+max per timestep) |",
        "| `(sfx)` | Softmax-weighted (sharpened inverse-MAE) |",
        "| `(geo)` | Sign-preserving geometric mean |",
        "| `(onl)` | Online adaptive (weights from previous window's error) |",
        "| `(ada)` | Adaptive Hedge — multiplicative weight update, optimal η=√(2 ln N/T) |",
        "| `(smo)` | Smooth online — EMA of inverse-MAE (blend of online and Hedge) |",
        "| `(sel)` | Model selection — greedy pick of single best model by EMA-MAE |",
        "| `(wh)`  | Per-horizon weights — distinct model mix per forecast step |",
        "| `(uq)`  | Uncertainty-weighted — IQR for Toto/Chronos, MAE-fallback for others |",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
        "1. [Individual Models — MAE](#individual-models--mae)",
        "2. [Individual Models — MASE](#individual-models--mase-scale-free)",
        "3. [Inference Latency](#inference-latency)",
        "4. [Best Ensemble vs Best Solo](#best-ensemble-vs-best-solo)",
        "5. [Ensemble Strategy Analysis](#ensemble-strategy-analysis)",
        "6. [Horizon Scaling](#horizon-scaling)",
        "7. [Context Length Scaling](#context-length-scaling)",
        "8. [Per-Dataset Detail](#per-dataset-detail)",
        "9. [Methodology](#methodology)",
        "",
        "---",
        "",
    ]

    # Section 1: MAE summary
    lines += [
        "## Individual Models — MAE",
        "",
        f"> context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, {args.main_windows} windows. "
        "**Bold** = best on that dataset.",
        "",
    ]
    rows = []
    for ds in DATASETS_ORDER:
        solo = main_results.get(ds, {}).get("solo", {})
        meta = DATASET_META[ds]
        maes = {m: solo[m]["MAE"] for m in SOLO_MODELS if m in solo}
        if not maes:
            continue
        best_v = min(maes.values())
        def cell(m, _maes=maes, _bv=best_v):
            if m not in _maes: return "—"
            v = fmt(_maes[m])
            return f"**{v}**" if _maes[m] == _bv else v
        rows.append([ds, meta["freq"], meta["domain"]] +
                    [cell(m) for m in SOLO_MODELS] +
                    [min(maes, key=maes.get)])
    lines.append(md_table(
        ["Dataset", "Freq", "Domain"] + [MODEL_LABELS[m] for m in SOLO_MODELS] + ["Winner"],
        rows))
    lines.append("")

    # Section 2: MASE summary
    lines += [
        "## Individual Models — MASE (scale-free)",
        "",
        "> MASE < 1 means the model beats a naïve 1-step random walk. **Bold** = best.",
        "",
    ]
    rows = []
    for ds in DATASETS_ORDER:
        solo  = main_results.get(ds, {}).get("solo", {})
        mases = {m: solo[m]["MASE"] for m in SOLO_MODELS if m in solo}
        if not mases: continue
        best_v = min(mases.values())
        def cell_m(m, _mases=mases, _bv=best_v):
            if m not in _mases: return "—"
            v = fmt(_mases[m], 3)
            return f"**{v}**" if _mases[m] == _bv else v
        rows.append([ds] + [cell_m(m) for m in SOLO_MODELS] +
                    [min(mases, key=mases.get)])
    lines.append(md_table(
        ["Dataset"] + [MODEL_LABELS[m] for m in SOLO_MODELS] + ["Winner"], rows))
    lines.append("")

    # Section 3: Latency
    lines += [
        "## Inference Latency",
        "",
        f"> Milliseconds per window (context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}).",
        "",
    ]
    rows = []
    for ds in DATASETS_ORDER:
        solo = main_results.get(ds, {}).get("solo", {})
        rows.append([ds] + [
            f"{solo[m]['lat_ms']:.0f}" if m in solo and solo[m]["lat_ms"] else "—"
            for m in SOLO_MODELS
        ])
    lines.append(md_table(["Dataset"] + [MODEL_LABELS[m] for m in SOLO_MODELS], rows))
    lines.append("")

    # Section 4: Best ensemble
    lines += [
        "## Best Ensemble vs Best Solo",
        "",
        "> Δ MAE = ensemble MAE − best solo MAE. Negative = ensemble wins.",
        "",
    ]
    rows = []
    for ds in DATASETS_ORDER:
        r    = main_results.get(ds, {})
        solo = r.get("solo", {})
        ens  = r.get("ensembles", [])
        if not solo or not ens: continue
        maes = {m: solo[m]["MAE"] for m in SOLO_MODELS if m in solo}
        best_solo_name = min(maes, key=maes.get)
        best_solo_mae  = maes[best_solo_name]
        best_ens = ens[0]
        delta    = best_ens["MAE"] - best_solo_mae
        pct      = abs(delta) / best_solo_mae * 100
        direction = f"↓ {pct:.1f}%" if delta < 0 else f"↑ {pct:.1f}%"
        rows.append([ds, best_solo_name, fmt(best_solo_mae),
                     best_ens["name"], fmt(best_ens["MAE"]),
                     f"{delta:+.4f}", direction])
    lines.append(md_table(
        ["Dataset","Best Solo","Solo MAE","Best Ensemble","Ensemble MAE","Δ MAE","Change"],
        rows))
    lines += ["", "---", ""]

    # Section 5: Ensemble Strategy Analysis
    lines += [
        "## Ensemble Strategy Analysis",
        "",
        "> Which strategies and model combinations win most often, and when do ensembles help?",
        "",
    ]

    _STRATEGY_LABELS = {
        "μ": "Mean (μ)", "~": "Median (~)", "w": "Inv-MAE weighted (w)",
        "tr": "Trimmed mean (tr)", "sfx": "Softmax (sfx)",
        "geo": "Geometric mean (geo)", "onl": "Online adaptive (onl)",
        "ada": "Hedge algorithm (ada)", "smo": "Smooth online (smo)",
        "sel": "Model selection (sel)", "wh": "Per-horizon (wh)",
        "uq": "Uncertainty-weighted (uq)",
    }
    _STRATEGIES_ORDER = ["w","onl","ada","smo","sel","wh","uq","~","geo","μ","tr","sfx"]
    _MODEL_DECODE = {"T":"Toto","C":"Chronos","F":"TimesFM","S":"Sundial",
                     "K":"TTM","L":"Lag-Llama","M":"Moment","O":"Moirai","P":"Moirai-2","W":"FlowState"}

    strategy_stats: dict = {}
    combo_stats:    dict = {}
    improve_list:   list = []
    hurt_list:      list = []

    for ds in DATASETS_ORDER:
        r    = main_results.get(ds, {})
        solo = r.get("solo", {})
        ens  = r.get("ensembles", [])
        if not solo or not ens: continue
        maes = {m: solo[m]["MAE"] for m in SOLO_MODELS if m in solo}
        if not maes: continue
        best_solo_mae = min(maes.values())
        best_ens = ens[0]
        delta    = best_ens["MAE"] - best_solo_mae
        norm_pct = delta / best_solo_mae * 100

        em = re.match(r"^([A-Z]+)\(([^)]+)\)$", best_ens["name"])
        if not em: continue
        combo, strategy = em.group(1), em.group(2)

        sc = strategy_stats.setdefault(strategy, {"times": 0, "norm_sum": 0.0, "improves": 0})
        sc["times"]    += 1
        sc["norm_sum"] += norm_pct
        if delta < 0: sc["improves"] += 1

        cc = combo_stats.setdefault(combo, {"times": 0, "improves": 0})
        cc["times"] += 1
        if delta < 0: cc["improves"] += 1

        (improve_list if delta < 0 else hurt_list).append((ds, norm_pct))

    lines += ["### Strategy Win Counts", "",
              "> 'Times Best' = datasets where this strategy appears in the overall best ensemble.",
              "> Avg Δ is normalized by best-solo MAE (negative = improvement).", ""]
    s_rows = []
    for s in _STRATEGIES_ORDER:
        sc = strategy_stats.get(s, {})
        t  = sc.get("times", 0)
        if t:
            avg = sc["norm_sum"] / t
            s_rows.append([_STRATEGY_LABELS[s], str(t), f"{avg:+.1f}%", f"{sc['improves']}/{t}"])
        else:
            s_rows.append([_STRATEGY_LABELS[s], "0", "—", "—"])
    lines.append(md_table(["Strategy","Times Best","Avg Δ (normalized)","Improves vs Solo"], s_rows))
    lines.append("")

    lines += ["### Model Combination Win Counts", "",
              "> T=Toto, C=Chronos, F=TimesFM, S=Sundial, K=TTM, L=Lag-Llama, M=Moment, "
              "O=Moirai, P=Moirai-2, W=FlowState.", ""]
    sorted_combos = sorted(combo_stats.items(), key=lambda x: -x[1]["times"])
    c_rows = []
    for combo, cc in sorted_combos:
        models_str = " + ".join(_MODEL_DECODE.get(c, c) for c in combo)
        c_rows.append([combo, models_str, str(cc["times"]), str(cc["improves"])])
    lines.append(md_table(["Combo","Models","Times Best","Improves vs Solo"], c_rows))
    lines.append("")

    n_imp = len(improve_list)
    n_hrt = len(hurt_list)
    improve_list.sort(key=lambda x: x[1])
    hurt_list.sort(key=lambda x: -x[1])

    lines += [
        "### When Ensembles Help vs Hurt", "",
        f"Ensembles improve accuracy on **{n_imp}/21** datasets and hurt on **{n_hrt}/21**.",
        "", "**Biggest improvements (normalized Δ MAE):**", "",
    ]
    lines.append(md_table(["Dataset","Δ MAE (normalized)"],
                           [(ds, f"{d:+.1f}%") for ds, d in improve_list[:5]]))
    lines += ["", "**Biggest regressions:**", ""]
    lines.append(md_table(["Dataset","Δ MAE (normalized)"],
                           [(ds, f"{d:+.1f}%") for ds, d in hurt_list[:5]]))
    lines += [
        "",
        "> **Key pattern:** Ensembles reliably hurt when one model is >2× better than the "
        "rest — there is no diversity to exploit. Median `(~)` is most reliable when "
        "applicable. CF (Chronos + TimesFM) is the most complementary pair.",
        "", "---", "",
    ]

    # Section 6: Horizon scaling
    lines += [
        "## Horizon Scaling",
        "",
        f"> context={MAIN_CONTEXT} fixed, horizons={HORIZONS}, {args.sweep_windows} windows.",
        "> **Bold** = best horizon for that dataset/model combination.",
        "> Trend column shows % change from h=96 to final valid horizon.",
        "",
        "### Average MASE across all datasets by horizon",
        "",
        "> Lower = better. Shows how each model degrades as the forecast horizon grows.",
        "",
    ]
    lines.append(build_avg_sweep_table(horizon_results, HORIZONS, "h", "MASE", 3))
    lines.append("")

    for model in SOLO_MODELS:
        lines += [f"### {MODEL_LABELS[model]} — MAE by horizon", ""]
        lines.append(build_sweep_table(horizon_results, HORIZONS, "h", "MAE", model, 4))
        lines.append("")

    lines += [
        "### Horizon scaling — MASE comparison (all models, key datasets)", "",
        "> One row per model × dataset. Shows forecast difficulty at longer horizons.", "",
    ]
    key_datasets = ["ETTh1", "ETTm1", "electricity", "exchange", "traffic"]
    for ds in key_datasets:
        lines += [f"**{ds}** ({DATASET_META[ds]['freq']}, {DATASET_META[ds]['domain']})", ""]
        h_rows = []
        for model in SOLO_MODELS:
            raw   = [horizon_results.get((ds, MAIN_CONTEXT, h), {}).get("solo", {})
                     .get(model, {}).get("MASE", float("nan")) for h in HORIZONS]
            cells = bold_min([fmt(r, 3) for r in raw], raw)
            h_rows.append([MODEL_LABELS[model]] + cells)
        lines.append(md_table(["Model"] + [f"h={h}" for h in HORIZONS], h_rows))
        lines.append("")

    lines += ["---", ""]

    # Section 7: Context length scaling
    lines += [
        "## Context Length Scaling",
        "",
        f"> horizon={MAIN_HORIZON} fixed, contexts={CONTEXTS}, {args.sweep_windows} windows.",
        "> **Bold** = best context length for that dataset/model combination.",
        "> Trend column shows % change from ctx=96 to ctx=1024.",
        "",
        "### Average MASE across all datasets by context length",
        "",
    ]
    lines.append(build_avg_sweep_table(context_results, CONTEXTS, "ctx", "MASE", 3))
    lines.append("")

    for model in SOLO_MODELS:
        lines += [f"### {MODEL_LABELS[model]} — MAE by context length", ""]
        lines.append(build_sweep_table(context_results, CONTEXTS, "ctx", "MAE", model, 4))
        lines.append("")

    lines += ["### Context scaling — MASE comparison (all models, key datasets)", ""]
    for ds in key_datasets:
        lines += [f"**{ds}** ({DATASET_META[ds]['freq']}, {DATASET_META[ds]['domain']})", ""]
        c_rows = []
        for model in SOLO_MODELS:
            raw   = [context_results.get((ds, ctx, MAIN_HORIZON), {}).get("solo", {})
                     .get(model, {}).get("MASE", float("nan")) for ctx in CONTEXTS]
            cells = bold_min([fmt(r, 3) for r in raw], raw)
            c_rows.append([MODEL_LABELS[model]] + cells)
        lines.append(md_table(["Model"] + [f"ctx={c}" for c in CONTEXTS], c_rows))
        lines.append("")

    lines += ["---", ""]

    # Section 8: Per-dataset detail
    lines += ["## Per-Dataset Detail", "",
              f"> context={MAIN_CONTEXT}, horizon={MAIN_HORIZON}, {args.main_windows} windows.", ""]

    for ds in DATASETS_ORDER:
        r    = main_results.get(ds, {})
        solo = r.get("solo", {})
        ens  = r.get("ensembles", [])
        meta = DATASET_META[ds]

        lines += [
            f"### {ds}", "",
            f"**Frequency:** {meta['freq']}  |  **Domain:** {meta['domain']}  |  "
            f"**Series length:** {r.get('series_len', 0):,}  |  "
            f"**Test rows:** {r.get('test_rows', 0):,}",
            "", "**Individual models:**", "",
        ]

        maes   = {m: solo[m]["MAE"] for m in SOLO_MODELS if m in solo}
        best_v = min(maes.values()) if maes else None
        solo_rows = []
        for m in SOLO_MODELS:
            if m not in solo: continue
            s   = solo[m]
            mae = fmt(s["MAE"])
            if s["MAE"] == best_v:
                mae = f"**{mae}**"
            solo_rows.append([m, mae, fmt(s["RMSE"]), fmt(s["MASE"], 3),
                               f"{s['lat_ms']:.0f}ms" if s["lat_ms"] else "—"])
        lines.append(md_table(["Model","MAE","RMSE","MASE","Latency"], solo_rows))
        lines.append("")

        if ens:
            lines += [f"**Top {TOP_N_ENSEMBLES} ensembles:**", ""]
            ens_rows = []
            for e in ens[:TOP_N_ENSEMBLES]:
                ens_rows.append([e["name"], fmt(e["MAE"]), fmt(e["RMSE"]),
                                 fmt(e["MASE"], 3), f"{e['delta']:+.4f}"])
            lines.append(md_table(["Ensemble","MAE","RMSE","MASE","Δ vs best solo"], ens_rows))
            lines.append("")

    # Section 9: Methodology
    lines += [
        "---", "",
        "## Methodology",
        "",
        "- All models run via Rust CLI binaries, F32 GGUF weights.",
        "- **Batch inference:** all rolling windows for a dataset are sent as a single "
          "batch request per model, so the binary and GGUF are loaded once per model "
          "rather than once per window.",
        "- Rolling windows evenly spaced over the held-out test split of each dataset.",
        "- **MAE** = mean absolute error averaged across all windows.",
        "- **RMSE** = root mean squared error averaged across windows.",
        "- **MASE** = MAE / mean-absolute-1-step-diff of the context window (scale-free).",
        "- Ensemble weights (`w`, `sfx`) derived from overall MAE on the same evaluation "
          "set (in-sample). `onl` weights update window-by-window from the previous "
          "window's per-model error.",
        "- **Horizon sweep:** context=512 fixed, horizons=[96, 192, 336, 720], "
          f"{args.sweep_windows} windows.",
        "- **Context sweep:** horizon=96 fixed, contexts=[96, 256, 512, 1024], "
          f"{args.sweep_windows} windows.",
        "- Datasets: ETDataset (ETTh/m), Monash (electricity, weather), "
          "Informer bundle (ILI, Exchange, Traffic), Jena Climate (Google TF).",
    ]

    md_text  = "\n".join(lines) + "\n"
    out_path = ROOT / "benchmark.md"
    out_path.write_text(md_text)

    print(f"\n{'='*60}")
    print(f"✓  Wrote {out_path}")
    print(f"   {len(lines):,} lines  |  {len(md_text):,} bytes")


if __name__ == "__main__":
    main()
