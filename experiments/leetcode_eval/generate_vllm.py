#!/usr/bin/env python3
"""Generate LeetCode solutions from a coding model served by vLLM's
OpenAI-compatible server, for every problem in newfacade/LeetCodeDataset.
This is the "positive code" / core-solving-ability generation step (design
doc Section 1 & 6).

Prerequisite (on the machine with a GPU):
    pip install vllm
    vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --port 8000

Then:
    python generate_vllm.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --output generations.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.dataset_utils import build_generation_prompt, extract_code, load_split
from common.model_client import ModelConfig, chat, make_client


def generate_one(client, cfg: ModelConfig, row: dict) -> dict:
    prompt = build_generation_prompt(row)
    raw, error = chat(client, cfg, prompt)
    return {
        "task_id": row["task_id"],
        "entry_point": row["entry_point"],
        "raw_response": raw,
        "generated_code": extract_code(raw) if raw else "",
        "error": error,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible server URL")
    ap.add_argument("--api-key", default="not-needed", help="vLLM ignores this but the client requires a value")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None, help="Only generate for the first N problems")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel in-flight requests to the server")
    ap.add_argument("--output", default="generations.jsonl")
    ap.add_argument("--resume", action="store_true", help="Skip task_ids already present in --output")
    args = ap.parse_args()

    ds = load_split(args.split, args.limit)
    rows = list(ds)

    done_ids = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["task_id"])
        print(f"Resuming: {len(done_ids)} problems already generated", file=sys.stderr)

    rows = [r for r in rows if r["task_id"] not in done_ids]
    if not rows:
        print("Nothing to generate.", file=sys.stderr)
        return

    cfg = ModelConfig(base_url=args.base_url, model=args.model, api_key=args.api_key,
                       temperature=args.temperature, max_tokens=args.max_tokens)
    client = make_client(cfg)

    mode = "a" if args.resume else "w"
    with out_path.open(mode) as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(generate_one, client, cfg, row): row for row in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generating"):
            result = fut.result()
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print(f"Wrote generations to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
