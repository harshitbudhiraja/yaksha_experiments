#!/usr/bin/env python3
"""Ask the model-under-test to inject a *specific* bug type into a known-good
reference solution, at a requested subtlety level, and also propose its own
fix. Design doc Section 2 (Bug-Injection Controllability) end to end.

Each (problem, bug_type, subtlety) combo is repeated `--repeats` times (with
temperature > 0) so evaluate.py can measure diversity/entropy across
generations for the same problem (Section 2.5).

Usage:
    python generate.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --split test --limit 30 --bug-types off_by_one,logic,edge_case_omission \
        --subtlety medium --repeats 3 --output bug_generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.bug_taxonomy import BUG_TYPES, BUG_TYPES_BY_ID
from common.dataset_utils import extract_all_code_blocks, get_reference_solution, load_split
from common.model_client import ModelConfig, chat, make_client

LINE_RE = re.compile(r"LINE:\s*(\d+|N/?A|none)", re.IGNORECASE)
DESC_RE = re.compile(r"DESCRIPTION:\s*(.+)")

SUBTLETY_HINT = {
    "easy": "Make the bug obvious and easy to spot on a careful read.",
    "medium": "Make the bug moderately subtle -- not obvious at a glance, but findable by tracing through an example.",
    "hard": "Make the bug very subtle -- it should look correct on a casual read and only fail on specific inputs.",
}


def build_prompt(reference: str, bug_type_id: str, subtlety: str) -> str:
    bug_type = BUG_TYPES_BY_ID[bug_type_id]
    return (
        "You are creating a debugging exercise for a coding student. You will "
        "be given a CORRECT reference solution. Introduce exactly ONE bug of "
        "the requested type, changing as little as possible, then describe it.\n\n"
        f"### Reference (correct) solution:\n```python\n{reference}\n```\n\n"
        f"### Requested bug type: {bug_type.label}\n{bug_type.request_instruction}\n\n"
        f"### Subtlety: {SUBTLETY_HINT[subtlety]}\n\n"
        "Respond in EXACTLY this format, with no extra commentary:\n\n"
        "### Buggy code\n```python\n<the mutated solution, same function signature>\n```\n"
        "### Bug info\n"
        "LINE: <the line number in the buggy code that contains the bug>\n"
        "DESCRIPTION: <one sentence describing the bug>\n"
        "### Suggested fix\n```python\n<the fully corrected solution -- i.e. undo just this bug>\n```\n"
    )


def parse_response(raw: str) -> dict:
    blocks = extract_all_code_blocks(raw)
    buggy_code = blocks[0] if len(blocks) >= 1 else ""
    fixed_code = blocks[1] if len(blocks) >= 2 else ""

    line_match = LINE_RE.search(raw)
    bug_line = None
    if line_match:
        val = line_match.group(1)
        bug_line = int(val) if val.isdigit() else None

    desc_match = DESC_RE.search(raw)
    bug_description = desc_match.group(1).strip() if desc_match else ""

    return {
        "buggy_code": buggy_code,
        "fixed_code": fixed_code,
        "bug_line_claimed": bug_line,
        "bug_description": bug_description,
        "parse_ok": bool(buggy_code),
    }


def generate_one(client, cfg: ModelConfig, row: dict, bug_type_id: str, subtlety: str, repeat_idx: int) -> dict:
    reference = get_reference_solution(row)
    prompt = build_prompt(reference, bug_type_id, subtlety)
    raw, error = chat(client, cfg, prompt)
    parsed = parse_response(raw) if raw else {
        "buggy_code": "", "fixed_code": "", "bug_line_claimed": None,
        "bug_description": "", "parse_ok": False,
    }
    return {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "bug_type_requested": bug_type_id,
        "subtlety": subtlety,
        "repeat_idx": repeat_idx,
        "raw_response": raw,
        "error": error,
        **parsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bug-types", default="all",
                     help="Comma-separated bug_taxonomy ids, or 'all' for every category")
    ap.add_argument("--subtlety", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per (problem, bug_type) for diversity/entropy")
    ap.add_argument("--temperature", type=float, default=0.8, help="Higher than solving-ability runs on purpose --diversity across repeats needs sampling variance")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default="bug_generations.jsonl")
    args = ap.parse_args()

    bug_type_ids = [b.id for b in BUG_TYPES] if args.bug_types == "all" else args.bug_types.split(",")
    for bid in bug_type_ids:
        if bid not in BUG_TYPES_BY_ID:
            sys.exit(f"Unknown bug type id: {bid}. Valid ids: {list(BUG_TYPES_BY_ID)}")

    ds = load_split(args.split, args.limit)
    rows = [r for r in ds if get_reference_solution(r)]
    skipped = len(ds) - len(rows)
    if skipped:
        print(f"Skipping {skipped} rows with no `completion` reference solution", file=sys.stderr)

    cfg = ModelConfig(base_url=args.base_url, model=args.model, api_key=args.api_key,
                       temperature=args.temperature, max_tokens=args.max_tokens)
    client = make_client(cfg)

    jobs = [
        (row, bug_type_id, args.subtlety, repeat_idx)
        for row in rows
        for bug_type_id in bug_type_ids
        for repeat_idx in range(args.repeats)
    ]

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_one, client, cfg, row, bt, subt, ri): (row, bt, ri)
                   for row, bt, subt, ri in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="injecting bugs"):
            out_f.write(json.dumps(fut.result()) + "\n")
            out_f.flush()

    print(f"Wrote {len(jobs)} generations to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
