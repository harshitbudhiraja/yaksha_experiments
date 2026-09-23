#!/usr/bin/env python3
"""Multi-turn code instruction-following (CodeIF-Bench, arXiv:2503.22688).

Complements `experiments/instruction_following/`, which scores single-shot
compliance with bug-injection meta-instructions. This one asks the harder
question: as a conversation accumulates verifiable requirements -- I/O
contract, exception behaviour, edge cases, type annotations, cyclomatic
complexity, PEP 8 -- does the model satisfy the *current* instruction without
silently dropping the ones it already satisfied?

Every instruction is verified by executing the benchmark's own unit tests, so
nothing here depends on a judge model.

Three interaction protocols (`--mode`):

  dynamic      The benchmark's adaptive protocol. Each round picks the first
               still-unsatisfied instruction, sends it alone, and on failure
               gives the model one feedback turn containing the error before
               moving on. Instructions that survive two failures are dropped.
  static       The instructions are sent one per turn in the benchmark's fixed
               `multi-turn` order, with no feedback and no retries. Isolates
               forgetting from error-driven repair.
  single_turn  Every instruction is stated up front in one prompt. The
               no-conversation control: whatever the model loses between
               `single_turn` and the multi-turn modes is a context-management
               loss, not a capability loss.

After each assistant turn, EVERY instruction in the task (including ones not
yet issued) is re-executed against the current code, so evaluate.py can derive
IA / CA / IFR / CIF -- and forgetting-by-age -- purely from this log.

Usage:
    ./fetch_data.sh                       # one-off: clone the benchmark data
    python run_dialogue.py \
        --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
        --mode dynamic --limit 20 --output mtif_dynamic.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from codeif_data import BASE_INSTRUCTION, CodeIFTask, load_tasks
from common.assert_runner import run_snippets
from common.dataset_utils import extract_all_code_blocks
from common.model_client import ModelConfig, chat_messages, make_client

FIRST_TURN_SUFFIX = (
    "\n\nRespond with the complete Python function in a single ```python fenced "
    "code block."
)

FOLLOW_UP_SUFFIX = (
    "\n\nRespond with the complete, updated Python function in a single ```python "
    "fenced code block. Keep every requirement from the earlier turns satisfied."
)


def _status_matrix(task: CodeIFTask, code: str, names: List[str],
                   timeout: int, concurrency: int) -> Dict[str, Dict[str, Any]]:
    """Execute every instruction's unit tests against `code`."""
    def _one(name: str):
        return name, run_snippets(code, task.tests_for(name), timeout=timeout)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return {name: res for name, res in pool.map(_one, names)}


def _extract_candidate(raw: str) -> Optional[str]:
    """The code a turn produced, or None if it produced none. A fenced block
    wins; failing that, bare text is accepted only if it actually parses."""
    blocks = [b for b in extract_all_code_blocks(raw) if b.strip()]
    if blocks:
        return blocks[0]
    stripped = (raw or "").strip()
    if stripped:
        try:
            compile(stripped, "<candidate>", "exec")
            return stripped
        except SyntaxError:
            pass
    return None


def _feedback_prompt(instruction: str, error: Optional[str]) -> str:
    msg = ("Your code does not satisfy this requirement yet:\n"
           f"\"{instruction}\"\n\n")
    if error:
        msg += f"The check failed with:\n```\n{error.strip()[-1500:]}\n```\n\n"
    return msg + ("Revise the function so this requirement holds, while keeping every "
                  "requirement from the earlier turns satisfied. Respond with a single "
                  "```python fenced code block.")


class _Dialogue:
    """One task's conversation: owns the message list, the model calls, the
    per-turn execution results, and the turn log every mode appends to."""

    def __init__(self, task: CodeIFTask, client, cfg: ModelConfig, args) -> None:
        self.task = task
        self.client = client
        self.cfg = cfg
        self.args = args
        self.messages: List[Dict[str, Any]] = []
        self.turns: List[Dict[str, Any]] = []
        self.code = ""
        self.names = task.issue_sequence()
        self.issued: List[str] = []

    def ask(self, prompt: str) -> Optional[str]:
        self.messages.append({"role": "user", "content": prompt})
        raw, error = chat_messages(self.client, self.cfg, self.messages)
        self.messages.append({"role": "assistant", "content": raw})
        new_code = _extract_candidate(raw)
        if new_code is not None:
            self.code = new_code
        # Otherwise keep the previous turn's code: a reply with nothing runnable
        # in it changed nothing, and scoring it as a total regression would
        # confuse "answered badly" with "forgot everything". Note this differs
        # from upstream, which falls back to treating the raw prose as code.
        return error

    def record(self, instruction_name: Optional[str], is_retry: bool, error: Optional[str]) -> Dict[str, Any]:
        status = _status_matrix(self.task, self.code, self.names,
                                self.args.test_timeout, self.args.test_concurrency)
        turn = {
            "turn": len(self.turns) + 1,
            "instruction_name": instruction_name,
            "is_retry": is_retry,
            "issued_so_far": list(self.issued),
            "code": self.code,
            "status": {n: s["pass"] for n, s in status.items()},
            "n_tests_passed": {n: [s["n_passed"], s["n_total"]] for n, s in status.items()},
            "first_error": (status[instruction_name]["first_error"] if instruction_name in status else None),
            "api_error": error,
        }
        self.turns.append(turn)
        return turn

    def log(self, mode: str) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "level": self.task.level,
            "mode": mode,
            "instruction_order": self.names,
            "turns": self.turns,
            "final_code": self.code,
            "dialogue": self.messages,
        }


def run_single_turn(task: CodeIFTask, client, cfg: ModelConfig, args) -> Dict[str, Any]:
    dlg = _Dialogue(task, client, cfg, args)
    numbered = "\n".join(
        f"{i + 1}. {task.instruction_for(n)}" for i, n in enumerate(task.turn_order))
    prompt = (f"{task.prompt}\n\nThe function must also satisfy all of the following "
              f"requirements:\n{numbered}{FIRST_TURN_SUFFIX}")
    dlg.issued = list(dlg.names)
    error = dlg.ask(prompt)
    dlg.record(BASE_INSTRUCTION, False, error)
    return dlg.log("single_turn")


def run_static(task: CodeIFTask, client, cfg: ModelConfig, args) -> Dict[str, Any]:
    dlg = _Dialogue(task, client, cfg, args)
    for i, name in enumerate(dlg.names):
        instruction = task.instruction_for(name)
        prompt = instruction + (FIRST_TURN_SUFFIX if i == 0 else FOLLOW_UP_SUFFIX)
        dlg.issued.append(name)
        error = dlg.ask(prompt)
        dlg.record(name, False, error)
    return dlg.log("static")


def run_dynamic(task: CodeIFTask, client, cfg: ModelConfig, args) -> Dict[str, Any]:
    dlg = _Dialogue(task, client, cfg, args)
    pending = list(dlg.names)
    max_turns = args.max_turns or (len(dlg.names) * 2 + 2)

    first = True
    while pending and len(dlg.turns) < max_turns:
        # Pick the first instruction the current code doesn't already satisfy.
        # Turn 1 always starts from the base prompt.
        if first:
            name = BASE_INSTRUCTION
        else:
            last_status = dlg.turns[-1]["status"]
            name = next((n for n in pending if not last_status.get(n)), None)
            if name is None:
                break  # everything issued and unissued alike already passes

        instruction = task.instruction_for(name)
        prompt = instruction + (FIRST_TURN_SUFFIX if first else FOLLOW_UP_SUFFIX)
        first = False
        if name not in dlg.issued:
            dlg.issued.append(name)
        error = dlg.ask(prompt)
        turn = dlg.record(name, False, error)

        if turn["status"].get(name):
            pending.remove(name)
            continue
        if len(dlg.turns) >= max_turns:
            break

        # One feedback turn quoting the failing check, then give up on it.
        error = dlg.ask(_feedback_prompt(instruction, turn["first_error"]))
        turn = dlg.record(name, True, error)
        pending.remove(name)

    return dlg.log("dynamic")


MODES = {"dynamic": run_dynamic, "static": run_static, "single_turn": run_single_turn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="not-needed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "CodeIF-Bench" / "data"))
    ap.add_argument("--level", default="L1", help="L1 = function level; L2/L3 need repo checkouts (see README)")
    ap.add_argument("--mode", default="dynamic", choices=sorted(MODES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-extension", action="store_true",
                    help="keep 'Functionality Extension' (changes the signature, so it "
                         "invalidates the base asserts -- off by default, as upstream)")
    ap.add_argument("--max-turns", type=int, default=None, help="dynamic mode only")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=4, help="tasks (i.e. dialogues) in flight")
    ap.add_argument("--test-concurrency", type=int, default=4, help="unit-test subprocesses per turn")
    ap.add_argument("--test-timeout", type=int, default=15)
    ap.add_argument("--reasoning-effort", default=None,
                    help="gpt-oss only: passed through as extra_body.reasoning_effort "
                         "(low/medium/high). Leaving it unset uses the server's default, "
                         "which for gpt-oss is high and roughly triples the run.")
    ap.add_argument("--output", default="mtif_dialogues.jsonl")
    args = ap.parse_args()

    tasks = load_tasks(args.data_dir, level=args.level, limit=args.limit,
                       include_extension=args.include_extension)

    cfg = ModelConfig(base_url=args.base_url, model=args.model, api_key=args.api_key,
                      temperature=args.temperature, max_tokens=args.max_tokens,
                      extra_body=({"reasoning_effort": args.reasoning_effort}
                                  if args.reasoning_effort else {}))
    client = make_client(cfg)
    runner = MODES[args.mode]
    write_lock = threading.Lock()

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(runner, task, client, cfg, args) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"multiturn-if/{args.mode}"):
            log = fut.result()
            with write_lock:
                out_f.write(json.dumps(log) + "\n")
                out_f.flush()

    print(f"Wrote {len(tasks)} dialogues to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
