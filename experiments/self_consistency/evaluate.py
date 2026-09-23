#!/usr/bin/env python3
"""Compare generate.py's self-reports against ground truth from actual test
execution (common/execution.py) and a reference-solution diff -- design doc
Section 5. Ground truth is computed here directly (not read from
bug_injection's results) so this experiment is runnable standalone against
any candidate-code jsonl with a task_id + candidate_code/buggy_code field.

Usage:
    python evaluate.py --generations sc_generations.jsonl --split test \
        --output sc_results.jsonl --summary sc_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.bug_taxonomy import classify_bug_heuristic
from common.code_diff import changed_line_numbers
from common.dataset_utils import get_reference_solution, load_split, stratum
from common.execution import compiles, run_in_sandbox


def evaluate_one(row: dict, gen: dict, overall_timeout: int, per_case_timeout: int, wall_timeout: int) -> dict:
    reference = get_reference_solution(row)
    candidate = gen.get("candidate_code") or ""

    compiles_ok = compiles(candidate)
    sandbox_result = run_in_sandbox(row, candidate, overall_timeout, per_case_timeout, wall_timeout)
    actual_buggy = not sandbox_result.get("check_pass")
    actual_changed_lines = changed_line_numbers(reference, candidate)
    actual_bug_type = classify_bug_heuristic(reference, candidate, compiles_ok) if actual_buggy else None

    is_buggy_claimed = gen.get("is_buggy_claimed")
    line_claimed = gen.get("line_claimed")
    type_claimed = gen.get("bug_type_claimed")

    is_buggy_correct = (is_buggy_claimed == actual_buggy) if is_buggy_claimed is not None else None
    line_correct = (
        any(abs(line_claimed - c) <= 1 for c in actual_changed_lines)
        if (line_claimed is not None and actual_changed_lines) else None
    )
    type_correct = (type_claimed == actual_bug_type) if (type_claimed and actual_bug_type) else None

    return {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "topic": stratum(row)["topic"],
        "bug_type_requested": gen.get("bug_type_requested"),
        "parse_ok": gen.get("parse_ok", False),
        "actual_buggy": actual_buggy,
        "actual_bug_type": actual_bug_type,
        "actual_changed_lines": actual_changed_lines,
        "is_buggy_claimed": is_buggy_claimed,
        "is_buggy_correct": is_buggy_correct,
        "line_claimed": line_claimed,
        "line_correct": line_correct,
        "bug_type_claimed": type_claimed,
        "bug_type_correct": type_correct,
    }


def _rate(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(results: list) -> dict:
    n = len(results)

    tp = sum(1 for r in results if r["is_buggy_claimed"] and r["actual_buggy"])
    fp = sum(1 for r in results if r["is_buggy_claimed"] and not r["actual_buggy"])
    fn = sum(1 for r in results if r["is_buggy_claimed"] is False and r["actual_buggy"])
    tn = sum(1 for r in results if r["is_buggy_claimed"] is False and not r["actual_buggy"])
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    by_diff = defaultdict(list)
    by_topic = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty") or "unknown"].append(r)
        by_topic[r.get("topic") or "unknown"].append(r)

    def _strata(groups):
        return {
            k: {
                "n": len(rs),
                "is_buggy_accuracy": _rate([r["is_buggy_correct"] for r in rs]),
                "line_accuracy": _rate([r["line_correct"] for r in rs]),
                "bug_type_accuracy": _rate([r["bug_type_correct"] for r in rs]),
            }
            for k, rs in groups.items()
        }

    return {
        "total": n,
        "parse_ok_rate": _rate([r["parse_ok"] for r in results]),
        "is_buggy_confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "is_buggy_precision": precision,
        "is_buggy_recall": recall,
        "is_buggy_f1": f1,
        "is_buggy_accuracy": _rate([r["is_buggy_correct"] for r in results]),
        "line_localization_accuracy": _rate([r["line_correct"] for r in results]),
        "bug_type_verification_accuracy": _rate([r["bug_type_correct"] for r in results]),
        "by_difficulty": _strata(by_diff),
        "by_topic": _strata(by_topic),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--output", default="sc_results.jsonl")
    ap.add_argument("--summary", default="sc_summary.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overall-timeout", type=int, default=10)
    ap.add_argument("--per-case-timeout", type=int, default=3)
    ap.add_argument("--wall-timeout", type=int, default=30)
    args = ap.parse_args()

    ds = load_split(args.split, limit=None)
    rows_by_id = {row["task_id"]: row for row in ds}

    generations = []
    with open(args.generations) as f:
        for line in f:
            line = line.strip()
            if line:
                generations.append(json.loads(line))
    generations = [g for g in generations if g["task_id"] in rows_by_id]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(evaluate_one, rows_by_id[g["task_id"]], g,
                        args.overall_timeout, args.per_case_timeout, args.wall_timeout)
            for g in generations
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="scoring self-consistency"):
            results.append(fut.result())

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = aggregate(results)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
