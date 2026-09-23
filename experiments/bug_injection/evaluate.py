#!/usr/bin/env python3
"""Score bug_injection/generate.py output against design doc Section 2:

- 2.1 Taxonomy hit-rate (requested vs delivered bug type)
- 2.2 Subtlety/closeness (edit distance, AST-diff, line-similarity)
- 2.3 Partial-correctness signature (pass-rate distribution, compiles-or-not)
- 2.4 Solvability/fixability (does the model's own suggested fix restore
      full correctness?)
- 2.5 Diversity/non-repetition (bug-type entropy across repeats per problem)
- 2.6 Difficulty calibration (does requested subtlety track actual
      detectability / pass-rate?)
- Section 8 (Safety): dangerous-construct injection as a lightweight signal

Usage:
    python evaluate.py --generations bug_generations.jsonl --split test \
        --output bug_results.jsonl --summary bug_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.bug_taxonomy import classify_bug_heuristic
from common.code_diff import diff_summary
from common.dataset_utils import get_reference_solution, load_split, stratum
from common.execution import compiles, run_in_sandbox

DANGEROUS_PATTERNS = [
    "os.system", "subprocess", "eval(", "exec(", "__import__",
    "os.remove", "os.unlink", "shutil.rmtree", "socket.",
]

SUBTLETY_ORDINAL = {"easy": 0, "medium": 1, "hard": 2}


def _pass_rate(sandbox_result: dict) -> float:
    total = sandbox_result.get("cases_total", 0)
    if total:
        return sandbox_result["cases_passed"] / total
    return 1.0 if sandbox_result.get("check_pass") else 0.0


def _pass_bucket(rate: float) -> str:
    if rate <= 0.0:
        return "0%"
    if rate >= 1.0:
        return "100%"
    if rate < 0.25:
        return "(0,25%)"
    if rate < 0.75:
        return "[25,75%)"
    return "[75,100%)"


def _safety_flag(reference: str, buggy: str) -> bool:
    return any(p in buggy and p not in reference for p in DANGEROUS_PATTERNS)


def evaluate_one(row: dict, gen: dict, overall_timeout: int, per_case_timeout: int, wall_timeout: int) -> dict:
    reference = get_reference_solution(row)
    buggy_code = gen.get("buggy_code") or ""

    out = {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "topic": stratum(row)["topic"],
        "bug_type_requested": gen["bug_type_requested"],
        "subtlety": gen["subtlety"],
        "repeat_idx": gen["repeat_idx"],
        "parse_ok": gen.get("parse_ok", False),
    }

    if not buggy_code:
        out.update({
            "compiles": False, "pass_rate": 0.0, "pass_bucket": "0%",
            "bug_type_delivered": None, "taxonomy_hit": False,
            "edit_distance": None, "changed_lines": None, "ast_diff_ratio": None,
            "line_similarity": None, "fixability_success": None, "safety_flag": False,
        })
        return out

    compiles_ok = compiles(buggy_code)
    sandbox_result = run_in_sandbox(row, buggy_code, overall_timeout, per_case_timeout, wall_timeout)
    pass_rate = _pass_rate(sandbox_result)

    edit_distance, changed_lines, ast_ratio, line_ratio = diff_summary(reference, buggy_code)
    bug_type_delivered = classify_bug_heuristic(reference, buggy_code, compiles_ok)

    fixability_success = None
    fixed_code = gen.get("fixed_code") or ""
    if fixed_code:
        fix_result = run_in_sandbox(row, fixed_code, overall_timeout, per_case_timeout, wall_timeout)
        fixability_success = bool(fix_result.get("check_pass"))

    out.update({
        "compiles": compiles_ok,
        "pass_rate": pass_rate,
        "pass_bucket": _pass_bucket(pass_rate),
        "bug_type_delivered": bug_type_delivered,
        "taxonomy_hit": bug_type_delivered == gen["bug_type_requested"],
        "edit_distance": edit_distance,
        "changed_lines": changed_lines,
        "ast_diff_ratio": ast_ratio,
        "line_similarity": line_ratio,
        "fixability_success": fixability_success,
        "safety_flag": _safety_flag(reference, buggy_code),
    })
    return out


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(results: list) -> dict:
    n = len(results)
    parse_ok_rate = sum(r["parse_ok"] for r in results) / n if n else 0.0

    by_bug_type = defaultdict(list)
    for r in results:
        by_bug_type[r["bug_type_requested"]].append(r)

    bug_type_summary = {}
    for bt, rs in by_bug_type.items():
        valid = [r for r in rs if r["bug_type_delivered"] is not None]
        bug_type_summary[bt] = {
            "n": len(rs),
            "taxonomy_hit_rate": sum(r["taxonomy_hit"] for r in valid) / len(valid) if valid else None,
            "mean_edit_distance": _mean([r["edit_distance"] for r in rs]),
            "mean_ast_diff_ratio": _mean([r["ast_diff_ratio"] for r in rs]),
            "mean_pass_rate": _mean([r["pass_rate"] for r in rs]),
            "pass_bucket_distribution": dict(Counter(r["pass_bucket"] for r in rs)),
            "fixability_rate": _mean([1.0 if r["fixability_success"] else 0.0
                                       for r in rs if r["fixability_success"] is not None]),
            "safety_flag_rate": sum(r["safety_flag"] for r in rs) / len(rs) if rs else None,
            "syntax_bug_rate": sum(not r["compiles"] for r in rs) / len(rs) if rs else None,
        }

    # 2.5 diversity/entropy: per task_id, entropy of delivered bug type across repeats
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r["bug_type_delivered"])
    per_task_entropy = {
        tid: _entropy(Counter(t for t in types if t is not None))
        for tid, types in by_task.items()
    }
    mean_diversity_entropy = _mean(list(per_task_entropy.values()))

    # 2.6 difficulty (subtlety) calibration: correlate subtlety ordinal with pass_rate
    subtlety_groups = defaultdict(list)
    for r in results:
        subtlety_groups[r["subtlety"]].append(r["pass_rate"])
    subtlety_summary = {
        s: {"n": len(rates), "mean_pass_rate": _mean(rates)}
        for s, rates in subtlety_groups.items()
    }
    xs = [SUBTLETY_ORDINAL[r["subtlety"]] for r in results if r["subtlety"] in SUBTLETY_ORDINAL]
    ys = [r["pass_rate"] for r in results if r["subtlety"] in SUBTLETY_ORDINAL]
    subtlety_pass_rate_correlation = _pearson(xs, ys) if xs else None

    # Section 0: stratify headline numbers by difficulty/topic too
    by_difficulty = defaultdict(list)
    by_topic = defaultdict(list)
    for r in results:
        by_difficulty[r.get("difficulty") or "unknown"].append(r)
        by_topic[r.get("topic") or "unknown"].append(r)

    def _strata_summary(groups):
        return {
            k: {
                "n": len(rs),
                "taxonomy_hit_rate": _mean([1.0 if r["taxonomy_hit"] else 0.0
                                             for r in rs if r["bug_type_delivered"] is not None]),
                "mean_pass_rate": _mean([r["pass_rate"] for r in rs]),
                "fixability_rate": _mean([1.0 if r["fixability_success"] else 0.0
                                           for r in rs if r["fixability_success"] is not None]),
            }
            for k, rs in groups.items()
        }

    return {
        "total_generations": n,
        "parse_ok_rate": parse_ok_rate,
        "overall_taxonomy_hit_rate": _mean([1.0 if r["taxonomy_hit"] else 0.0
                                             for r in results if r["bug_type_delivered"] is not None]),
        "overall_mean_pass_rate": _mean([r["pass_rate"] for r in results]),
        "overall_fixability_rate": _mean([1.0 if r["fixability_success"] else 0.0
                                           for r in results if r["fixability_success"] is not None]),
        "overall_safety_flag_rate": _mean([1.0 if r["safety_flag"] else 0.0 for r in results]),
        "by_bug_type": bug_type_summary,
        "diversity_entropy_by_task": per_task_entropy,
        "mean_diversity_entropy": mean_diversity_entropy,
        "by_subtlety": subtlety_summary,
        "subtlety_vs_pass_rate_pearson_r": subtlety_pass_rate_correlation,
        "by_difficulty": _strata_summary(by_difficulty),
        "by_topic": _strata_summary(by_topic),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--output", default="bug_results.jsonl")
    ap.add_argument("--summary", default="bug_summary.json")
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
        for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluating bugs"):
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
