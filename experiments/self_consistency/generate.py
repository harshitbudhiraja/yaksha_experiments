#!/usr/bin/env python3
"""Design doc Section 5 (Self-Consistency & Ground-Truth Reliability): can
the model correctly self-report whether code is buggy, and where, when
reviewing it blind (i.e. not told it's reviewing its own output)?

Takes bug_injection/generate.py's output as input (candidate buggy/correct
code to review) and asks the model to audit each one from scratch.

Usage:
    python generate.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --bug-generations ../bug_injection/bug_generations.jsonl \
        --split test --output sc_generations.jsonl
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

from common.bug_taxonomy import BUG_TYPES
from common.dataset_utils import load_split
from common.model_client import ModelConfig, chat, make_client

IS_BUGGY_RE = re.compile(r"IS_BUGGY:\s*(yes|no)", re.IGNORECASE)
LINE_RE = re.compile(r"LINE:\s*(\d+|N/?A|none)", re.IGNORECASE)
TYPE_RE = re.compile(r"TYPE:\s*(\w+)", re.IGNORECASE)
DESC_RE = re.compile(r"DESCRIPTION:\s*(.+)")

_TAXONOMY_LIST = "\n".join(f"- {b.id}: {b.description}" for b in BUG_TYPES)


def build_prompt(problem_description: str, candidate_code: str) -> str:
    return (
        "You are reviewing a candidate solution to a coding problem, submitted "
        "by a student. Determine whether it is correct or contains a bug.\n\n"
        f"### Problem:\n{problem_description}\n\n"
        f"### Candidate solution:\n```python\n{candidate_code}\n```\n\n"
        f"If it contains a bug, classify it using this taxonomy:\n{_TAXONOMY_LIST}\n\n"
        "Respond in EXACTLY this format, with no extra commentary:\n"
        "IS_BUGGY: <yes|no>\n"
        "TYPE: <taxonomy id, or none if not buggy>\n"
        "LINE: <line number of the bug, or none>\n"
        "DESCRIPTION: <one sentence, or 'looks correct' if not buggy>\n"
    )


def parse_response(raw: str) -> dict:
    is_buggy_match = IS_BUGGY_RE.search(raw)
    line_match = LINE_RE.search(raw)
    type_match = TYPE_RE.search(raw)
    desc_match = DESC_RE.search(raw)

    is_buggy_claimed = is_buggy_match.group(1).lower() == "yes" if is_buggy_match else None
    line_claimed = int(line_match.group(1)) if line_match and line_match.group(1).isdigit() else None
    type_claimed = type_match.group(1).lower() if type_match else None
    if type_claimed == "none":
        type_claimed = None

    return {
        "is_buggy_claimed": is_buggy_claimed,
        "bug_type_claimed": type_claimed,
        "line_claimed": line_claimed,
        "description_claimed": desc_match.group(1).strip() if desc_match else "",
        "parse_ok": is_buggy_match is not None,
    }


def generate_one(client, cfg: ModelConfig, row: dict, bug_gen: dict) -> dict:
    candidate_code = bug_gen.get("buggy_code") or ""
    prompt = build_prompt(row.get("problem_description", ""), candidate_code)
    raw, error = chat(client, cfg, prompt)
    parsed = parse_response(raw) if raw else {
        "is_buggy_claimed": None, "bug_type_claimed": None, "line_claimed": None,
        "description_claimed": "", "parse_ok": False,
    }
    return {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "bug_type_requested": bug_gen.get("bug_type_requested"),
        "subtlety": bug_gen.get("subtlety"),
        "repeat_idx": bug_gen.get("repeat_idx"),
        "candidate_code": candidate_code,
        "raw_response": raw,
        "error": error,
        **parsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bug-generations", required=True,
                     help="jsonl produced by experiments/bug_injection/generate.py")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="Only review the first N candidate generations")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default="sc_generations.jsonl")
    args = ap.parse_args()

    ds = load_split(args.split, limit=None)
    rows_by_id = {row["task_id"]: row for row in ds}

    bug_gens = []
    with open(args.bug_generations) as f:
        for line in f:
            line = line.strip()
            if line:
                bug_gens.append(json.loads(line))
    bug_gens = [g for g in bug_gens if g["task_id"] in rows_by_id and g.get("buggy_code")]
    if args.limit is not None:
        bug_gens = bug_gens[: args.limit]

    cfg = ModelConfig(base_url=args.base_url, model=args.model, api_key=args.api_key,
                       temperature=args.temperature, max_tokens=args.max_tokens)
    client = make_client(cfg)

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, cfg, rows_by_id[g["task_id"]], g) for g in bug_gens]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="self-review"):
            out_f.write(json.dumps(fut.result()) + "\n")
            out_f.flush()

    print(f"Wrote {len(bug_gens)} generations to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
