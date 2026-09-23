#!/usr/bin/env python3
"""Delta table between a base-model lm-eval run and a tuned-model (Yaksha)
run -- design doc Section 1: "did pedagogical alignment cost solving
ability?" (pedRL's headline comparison, e.g. SFT: -7.5% GSM8K, -9.4% MATH500).

Usage:
    python compare_base_vs_tuned.py --base results/base --tuned results/tuned
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _latest_results_json(results_dir: Path) -> Path:
    candidates = sorted(results_dir.rglob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No results_*.json found under {results_dir} -- run run_lm_eval.sh first"
        )
    return candidates[-1]


def _load_metrics(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """{task_name: {metric_name: value}}, keeping only numeric, non-stderr
    metrics (lm-eval keys look like 'pass@1,create_test' or 'acc,none')."""
    with open(_latest_results_json(results_dir)) as f:
        data = json.load(f)
    out: Dict[str, Dict[str, float]] = {}
    for task, metrics in data.get("results", {}).items():
        out[task] = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and "stderr" not in k and k != "alias"
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="lm_eval --output_path dir for the base model")
    ap.add_argument("--tuned", required=True, help="lm_eval --output_path dir for the tuned (Yaksha) model")
    ap.add_argument("--output", default="base_vs_tuned.json")
    args = ap.parse_args()

    base_metrics = _load_metrics(Path(args.base))
    tuned_metrics = _load_metrics(Path(args.tuned))

    rows = []
    for task in sorted(set(base_metrics) | set(tuned_metrics)):
        b = base_metrics.get(task, {})
        t = tuned_metrics.get(task, {})
        for metric in sorted(set(b) | set(t)):
            b_val, t_val = b.get(metric), t.get(metric)
            delta = (t_val - b_val) if (b_val is not None and t_val is not None) else None
            rows.append({
                "task": task, "metric": metric,
                "base": b_val, "tuned": t_val, "delta": delta,
            })

    print(f"{'task':<20} {'metric':<20} {'base':>10} {'tuned':>10} {'delta':>10}")
    for r in rows:
        b_str = f"{r['base']:.4f}" if r["base"] is not None else "n/a"
        t_str = f"{r['tuned']:.4f}" if r["tuned"] is not None else "n/a"
        d_str = f"{r['delta']:+.4f}" if r["delta"] is not None else "n/a"
        print(f"{r['task']:<20} {r['metric']:<20} {b_str:>10} {t_str:>10} {d_str:>10}")

    with open(args.output, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
