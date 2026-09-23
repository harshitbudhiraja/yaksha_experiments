#!/usr/bin/env python3
"""Convert CodeIF-Bench upstream run logs into this experiment's dialogue-log
schema, so evaluate.py can score runs that were produced by the authors'
harness (e.g. the August runs under ~/evaluation/codeif-eval/).

The upstream logs carry the per-turn code but only the metrics upstream chose
to compute. evaluate.py derives everything from a per-turn *status matrix*
(every instruction re-executed against every turn's code), so this script
replays the stored code through the same unit tests. No model is called --
this is pure re-scoring of finished generations.

Both upstream layouts are handled:
  dynamic  turns[].code plus turns[].results[0].instruction_type
  static   session_messages, assistant turns in `multi-turn` order

`Functionality Extension` is dropped from both (it changes the signature, so it
invalidates the base asserts). Upstream's dynamic script already excludes it;
its static script doesn't, so the trailing static turn that introduces it is
dropped here to keep the two modes comparable.

Usage:
    python import_upstream.py --upstream <file.jsonl> --mode dynamic \
        --data-dir CodeIF-Bench/data --output imported.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from codeif_data import BASE_INSTRUCTION, EXTENSION_TYPE, CodeIFTask, load_tasks
from common.assert_runner import run_snippets
from common.dataset_utils import extract_all_code_blocks


def _read(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _extract(raw: str) -> str:
    blocks = [b for b in extract_all_code_blocks(raw or "") if b.strip()]
    return blocks[0] if blocks else (raw or "").strip()


def _dynamic_turns(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """(instruction_name, code) per turn, in order."""
    out = []
    for t in entry.get("turns") or []:
        results = t.get("results") or [{}]
        out.append({"name": results[0].get("instruction_type"), "code": t.get("code") or ""})
    return out


def _static_turns(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    order = [BASE_INSTRUCTION] + [k for k in (entry.get("multi-turn") or [])]
    codes = [m["content"] for m in (entry.get("session_messages") or [])
             if m.get("role") == "assistant"]
    return [{"name": n, "code": _extract(c)} for n, c in zip(order, codes)]


def convert_one(entry: Dict[str, Any], task: CodeIFTask, mode: str,
                timeout: int, concurrency: int) -> Dict[str, Any]:
    raw_turns = _dynamic_turns(entry) if mode == "dynamic" else _static_turns(entry)
    raw_turns = [t for t in raw_turns if t["name"] != EXTENSION_TYPE]

    names = task.issue_sequence()  # Base + turn_order, extension already excluded
    turns: List[Dict[str, Any]] = []
    issued: List[str] = []
    prev_name = None

    for i, rt in enumerate(raw_turns):
        name = rt["name"]
        if name and name not in issued:
            issued.append(name)

        def _one(n: str):
            return n, run_snippets(rt["code"], task.tests_for(n), timeout=timeout)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            status = {n: res for n, res in pool.map(_one, names)}

        turns.append({
            "turn": i + 1,
            "instruction_name": name,
            # Upstream doesn't flag retries; a repeated instruction on
            # consecutive turns is one (that is exactly its retry protocol).
            "is_retry": bool(mode == "dynamic" and name == prev_name),
            "issued_so_far": list(issued),
            "code": rt["code"],
            "status": {n: s["pass"] for n, s in status.items()},
            "n_tests_passed": {n: [s["n_passed"], s["n_total"]] for n, s in status.items()},
            "first_error": status.get(name, {}).get("first_error") if name else None,
            "api_error": None,
        })
        prev_name = name

    return {
        "task_id": task.task_id,
        "level": task.level,
        "mode": mode,
        "instruction_order": names,
        "turns": turns,
        "final_code": turns[-1]["code"] if turns else "",
        "dialogue": entry.get("dialogue") or entry.get("session_messages") or [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--mode", required=True, choices=["dynamic", "static"])
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "CodeIF-Bench" / "data"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--test-timeout", type=int, default=15)
    ap.add_argument("--test-concurrency", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=6, help="tasks converted in parallel")
    args = ap.parse_args()

    tasks = {t.task_id: t for t in load_tasks(args.data_dir, limit=None)}
    entries = _read(args.upstream)

    jobs = [(e, tasks[str(e.get("task_id"))]) for e in entries
            if str(e.get("task_id")) in tasks]
    missing = len(entries) - len(jobs)
    if missing:
        print(f"warning: {missing} entries had no matching task in {args.data_dir}", file=sys.stderr)

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(convert_one, e, t, args.mode, args.test_timeout, args.test_concurrency)
                   for e, t in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"import/{args.mode}"):
            out_f.write(json.dumps(fut.result()) + "\n")
            out_f.flush()

    print(f"Wrote {len(jobs)} dialogues to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
