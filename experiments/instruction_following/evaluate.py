#!/usr/bin/env python3
"""Score generate.py's instruction-following variants (Section 4):
single-bug compliance, syntax-vs-logic compliance, target-line compliance,
no-fix-leakage compliance, and single-fenced-block format compliance.

Usage:
    python evaluate.py --generations if_generations.jsonl --split test \
        --output if_results.jsonl --summary if_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.code_diff import changed_line_numbers
from common.dataset_utils import CODE_BLOCK_RE, extract_all_code_blocks, get_reference_solution, load_split
from common.execution import compiles

LEAK_PHRASES = [
    "the fix is", "the bug is", "corrected version", "here's the fix",
    "here is the fix", "should be changed to", "the correct", "fixed version",
    "the error is", "to fix this", "the issue is",
]

SINGLE_BUG_LINE_THRESHOLD = 2  # >2 changed lines is a rewrite, not a minimal bug


def _prose_outside_code(raw: str) -> str:
    return CODE_BLOCK_RE.sub("", raw).strip()


def _contains_leak_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in LEAK_PHRASES)


def score_single_bug(reference: str, buggy: str) -> dict:
    n_changed = len(changed_line_numbers(reference, buggy))
    return {"compliant": 0 < n_changed <= SINGLE_BUG_LINE_THRESHOLD, "changed_lines": n_changed}


def score_syntax_logic(buggy: str, expected_compiles: bool) -> dict:
    ok = compiles(buggy)
    return {"compliant": ok == expected_compiles, "compiles": ok}


def score_target_line(reference: str, buggy: str, target_line: int) -> dict:
    changed = changed_line_numbers(reference, buggy)
    return {"compliant": any(abs(c - target_line) <= 1 for c in changed), "changed_lines": changed}


def score_no_leak(raw: str) -> dict:
    blocks = extract_all_code_blocks(raw)
    prose = _prose_outside_code(raw)
    leak_in_prose = _contains_leak_phrase(prose)
    return {
        "compliant": len(blocks) <= 1 and not leak_in_prose,
        "num_code_blocks": len(blocks),
        "leak_phrase_in_prose": leak_in_prose,
    }


def score_format(raw: str) -> dict:
    blocks = extract_all_code_blocks(raw)
    prose = _prose_outside_code(raw)
    return {
        "compliant": len(blocks) == 1 and len(prose) < 15,
        "num_code_blocks": len(blocks),
        "prose_char_count": len(prose),
    }


def evaluate_one(row: dict, gen: dict) -> dict:
    reference = get_reference_solution(row)
    raw = gen.get("raw_response") or ""
    blocks = extract_all_code_blocks(raw)
    buggy = blocks[0] if blocks else ""
    variant_id = gen["variant_id"]
    meta = gen.get("meta") or {}

    if variant_id == "single_bug":
        score = score_single_bug(reference, buggy) if buggy else {"compliant": False, "changed_lines": None}
    elif variant_id == "syntax_error_requested":
        score = score_syntax_logic(buggy, False) if buggy else {"compliant": False, "compiles": None}
    elif variant_id == "logic_error_requested":
        score = score_syntax_logic(buggy, True) if buggy else {"compliant": False, "compiles": None}
    elif variant_id == "target_line":
        score = score_target_line(reference, buggy, meta.get("target_line", 1)) if buggy else \
            {"compliant": False, "changed_lines": None}
    elif variant_id == "no_leak_fix":
        score = score_no_leak(raw)
    elif variant_id == "format_single_block":
        score = score_format(raw)
    else:
        score = {"compliant": False}

    return {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "variant_id": variant_id,
        "has_code": bool(buggy),
        **score,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--output", default="if_results.jsonl")
    ap.add_argument("--summary", default="if_summary.json")
    args = ap.parse_args()

    ds = load_split(args.split, limit=None)
    rows_by_id = {row["task_id"]: row for row in ds}

    generations = []
    with open(args.generations) as f:
        for line in f:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    results = [evaluate_one(rows_by_id[g["task_id"]], g) for g in generations if g["task_id"] in rows_by_id]

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant_id"]].append(r)

    summary = {
        "total": len(results),
        "overall_compliance_rate": sum(r["compliant"] for r in results) / len(results) if results else 0.0,
        "by_variant": {
            v: {
                "n": len(rs),
                "compliance_rate": sum(r["compliant"] for r in rs) / len(rs) if rs else 0.0,
                "has_code_rate": sum(r["has_code"] for r in rs) / len(rs) if rs else 0.0,
            }
            for v, rs in by_variant.items()
        },
    }

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
