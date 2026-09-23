#!/usr/bin/env python3
"""Design doc Section 4 (Instruction Following), scoped to the
meta-instructions specific to bug-injection: "inject exactly one bug",
"make it a runtime/syntax error not a logic error", "target this specific
line", "don't leak the fix", plus basic format compliance.

Six instruction variants are sent per problem; evaluate.py scores each
automatically against what a compliant response must look like.

Usage:
    python generate.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --split test --limit 30 --output if_generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.dataset_utils import get_reference_solution, load_split
from common.model_client import ModelConfig, chat, make_client


def _numbered(code: str) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(code.splitlines()))


def _pick_target_line(reference: str) -> int:
    lines = reference.splitlines()
    non_trivial = [i + 1 for i, l in enumerate(lines) if l.strip() and not l.strip().startswith("#")]
    return non_trivial[len(non_trivial) // 2] if non_trivial else 1


def build_variants(reference: str) -> list:
    target_line = _pick_target_line(reference)
    return [
        {
            "variant_id": "single_bug",
            "meta": {},
            "prompt": (
                "You are creating a debugging exercise. Given the correct solution "
                "below, introduce EXACTLY ONE bug -- change as little as possible, "
                "everything else must stay identical. Respond with only a single "
                "```python fenced code block containing the mutated solution.\n\n"
                f"```python\n{reference}\n```"
            ),
        },
        {
            "variant_id": "syntax_error_requested",
            "meta": {"expected_compiles": False},
            "prompt": (
                "Given the correct solution below, introduce a bug that causes a "
                "SYNTAX error -- the result must fail to parse/compile. Do NOT "
                "introduce a logic error instead. Respond with only a single "
                "```python fenced code block.\n\n"
                f"```python\n{reference}\n```"
            ),
        },
        {
            "variant_id": "logic_error_requested",
            "meta": {"expected_compiles": True},
            "prompt": (
                "Given the correct solution below, introduce a LOGIC error only -- "
                "the result MUST still parse/compile without any syntax error. Do "
                "NOT introduce a syntax error. Respond with only a single "
                "```python fenced code block.\n\n"
                f"```python\n{reference}\n```"
            ),
        },
        {
            "variant_id": "target_line",
            "meta": {"target_line": target_line},
            "prompt": (
                f"Given the correct, line-numbered solution below, introduce a bug "
                f"specifically on line {target_line} (do not change any other line). "
                "Respond with only a single ```python fenced code block containing "
                "the mutated solution WITHOUT the line-number prefixes.\n\n"
                f"```\n{_numbered(reference)}\n```"
            ),
        },
        {
            "variant_id": "no_leak_fix",
            "meta": {},
            "prompt": (
                "Given the correct solution below, introduce exactly one bug. Output "
                "ONLY the buggy code. Do NOT reveal, describe, explain, or hint at "
                "what the bug is or how to fix it anywhere in your response -- no "
                "second code block, no comments about the bug, no prose at all. "
                "Respond with only a single ```python fenced code block.\n\n"
                f"```python\n{reference}\n```"
            ),
        },
        {
            "variant_id": "format_single_block",
            "meta": {},
            "prompt": (
                "Given the correct solution below, introduce exactly one bug. Your "
                "ENTIRE response must be a single ```python fenced code block and "
                "nothing else -- no explanation before or after it.\n\n"
                f"```python\n{reference}\n```"
            ),
        },
    ]


def generate_one(client, cfg: ModelConfig, row: dict, variant: dict) -> dict:
    raw, error = chat(client, cfg, variant["prompt"])
    return {
        "task_id": row["task_id"],
        "difficulty": row.get("difficulty"),
        "variant_id": variant["variant_id"],
        "meta": variant["meta"],
        "raw_response": raw,
        "error": error,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--output", default="if_generations.jsonl")
    args = ap.parse_args()

    ds = load_split(args.split, args.limit)
    rows = [r for r in ds if get_reference_solution(r)]

    cfg = ModelConfig(base_url=args.base_url, model=args.model, api_key=args.api_key,
                       temperature=args.temperature, max_tokens=args.max_tokens)
    client = make_client(cfg)

    jobs = [(row, variant) for row in rows for variant in build_variants(get_reference_solution(row))]

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_one, client, cfg, row, variant) for row, variant in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="instruction-following"):
            out_f.write(json.dumps(fut.result()) + "\n")
            out_f.flush()

    print(f"Wrote {len(jobs)} generations to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
