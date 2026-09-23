#!/usr/bin/env python3
"""Evaluate model-generated LeetCode solutions (from generate_vllm.py)
against newfacade/LeetCodeDataset's tests, in an isolated, resource-limited
subprocess per problem. This is the "positive code" / core-solving-ability
metric (design doc Section 1 & 6): pass@1 and test-case pass rate.

Usage:
    python evaluate.py --generations generations.jsonl --output results.jsonl
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

from common.dataset_utils import load_split, stratum
from common.execution import run_in_sandbox


def evaluate_one(row: dict, generated_code: str, overall_timeout: int, per_case_timeout: int,
                  wall_timeout: int) -> dict:
    result = run_in_sandbox(row, generated_code, overall_timeout, per_case_timeout, wall_timeout)
    result["difficulty"] = row.get("difficulty")
    result["topic"] = stratum(row)["topic"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True, help="JSONL file produced by generate_vllm.py")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--summary", default="summary.json")
    ap.add_argument("--workers", type=int, default=8, help="Parallel sandbox subprocesses")
    ap.add_argument("--overall-timeout", type=int, default=10, help="Seconds allowed for the whole check() call")
    ap.add_argument("--per-case-timeout", type=int, default=3, help="Seconds allowed per input_output case")
    ap.add_argument("--wall-timeout", type=int, default=30, help="Hard subprocess kill timeout")
    args = ap.parse_args()

    ds = load_split(args.split, limit=None)
    rows_by_id = {row["task_id"]: row for row in ds}

    generations = []
    with open(args.generations) as f:
        for line in f:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    missing = [g["task_id"] for g in generations if g["task_id"] not in rows_by_id]
    if missing:
        print(f"WARNING: {len(missing)} generation task_ids not found in dataset split, skipping", file=sys.stderr)
    generations = [g for g in generations if g["task_id"] in rows_by_id]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_one,
                rows_by_id[g["task_id"]],
                g.get("generated_code", ""),
                args.overall_timeout,
                args.per_case_timeout,
                args.wall_timeout,
            ): g["task_id"]
            for g in generations
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluating"):
            results.append(fut.result())

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total_problems = len(results)
    problems_passed = sum(1 for r in results if r["check_pass"])
    total_cases = sum(r["cases_total"] for r in results)
    cases_passed = sum(r["cases_passed"] for r in results)

    by_difficulty = defaultdict(lambda: {"total": 0, "passed": 0})
    by_topic = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        d = r.get("difficulty") or "unknown"
        by_difficulty[d]["total"] += 1
        by_difficulty[d]["passed"] += 1 if r["check_pass"] else 0
        t = r.get("topic") or "unknown"
        by_topic[t]["total"] += 1
        by_topic[t]["passed"] += 1 if r["check_pass"] else 0

    def _rate(v):
        return {**v, "rate": v["passed"] / v["total"] if v["total"] else 0.0}

    summary = {
        "model_generations_file": args.generations,
        "total_problems": total_problems,
        "problems_passed_pass@1": problems_passed,
        "pass@1_rate": problems_passed / total_problems if total_problems else 0.0,
        "total_test_cases": total_cases,
        "test_cases_passed": cases_passed,
        "test_case_pass_rate": cases_passed / total_cases if total_cases else 0.0,
        "by_difficulty": {k: _rate(v) for k, v in by_difficulty.items()},
        "by_topic": {k: _rate(v) for k, v in by_topic.items()},
    }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
