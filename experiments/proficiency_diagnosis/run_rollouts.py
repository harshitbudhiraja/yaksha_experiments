#!/usr/bin/env python3
"""Proficiency diagnosis: how good is a tutor LLM at recovering a student's
latent 12-stage proficiency vector from dialogue?

One episode = one (ground-truth profile, problem) pair. A simulated student is
conditioned on P_true and makes a first attempt at the problem; the tutor then
runs a Probe -> Observe -> Update loop for up to N turns, emitting a full belief
vector every turn. The final belief is P_pred. Nothing in the tutor's context
ever contains P_true.

Three conditions, all scored the same way by evaluate.py:
    interactive  full probing loop (the actual experiment)
    one_shot     tutor sees only the student's first attempt, then must score
                 (isolates the value of *probing* vs. just reading code)
    blind        tutor never sees the student at all (prior-only floor; if
                 `interactive` doesn't beat this, either the student isn't
                 expressing the vector or the tutor isn't using the dialogue)

Usage:
    python run_rollouts.py \
        --tutor-model gpt-oss-20b --tutor-base-url http://localhost:8000/v1 \
        --student-model qwen2.5-7b --student-base-url http://localhost:8001/v1 \
        --turns 6 --problems 3 --profiles-per-archetype 2 \
        --output rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from common.dataset_utils import load_split
from common.model_client import ModelConfig, make_client
from common.stages import STAGE_IDS

from profiles import ARCHETYPES, sample_n_profiles, sample_profiles
from student import build_first_task, student_reply
from tutor import tutor_turn

CONDITIONS = ["interactive", "one_shot", "blind"]


def problem_text(row: Dict[str, Any]) -> str:
    desc = row.get("problem_description") or ""
    starter = row.get("starter_code") or ""
    return f"{desc}\n\nStarter code:\n```python\n{starter}\n```" if starter else desc


def run_episode(tutor_client, tutor_cfg: ModelConfig,
                student_client, student_cfg: ModelConfig,
                row: Dict[str, Any], prof: Dict[str, Any],
                turns: int, condition: str, honor_stop: bool) -> Dict[str, Any]:
    ptext = problem_text(row)
    profile = prof["profile"]

    exchanges: List[Dict[str, str]] = []
    turn_log: List[Dict[str, Any]] = []
    errors: List[str] = []

    if condition != "blind":
        first_task = build_first_task(ptext)
        reply, err = student_reply(student_client, student_cfg, profile, [], first_task)
        if err:
            errors.append(f"student(turn0): {err}")
        exchanges.append({"tutor": first_task, "student": reply,
                          "source": "fixed_first_attempt"})

    budget = 1 if condition in ("one_shot", "blind") else turns
    last_belief = None

    for t in range(1, budget + 1):
        is_final = (t == budget)
        action, raw, perr = tutor_turn(
            tutor_client, tutor_cfg, ptext, exchanges, t, budget,
            is_final=is_final, blind=(condition == "blind"))

        if perr:
            errors.append(f"tutor(turn{t}): {perr}")
        belief = action["belief"] if action else last_belief
        entry = {
            "turn": t,
            "parse_ok": action is not None,
            "parse_error": perr,
            "belief": belief,
            "confidence": action["confidence"] if action else None,
            "target_stages": action["target_stages"] if action else [],
            "probe": action["probe"] if action else "",
            "reasoning": action["reasoning"] if action else "",
            "declared_stop": bool(action["stop"]) if action else False,
            "raw_tutor_response": raw,
        }
        turn_log.append(entry)
        if belief is not None:
            last_belief = belief

        if is_final:
            break
        if entry["declared_stop"] and honor_stop:
            break
        if not entry["probe"]:
            errors.append(f"tutor(turn{t}): empty probe, ending episode")
            break

        reply, serr = student_reply(student_client, student_cfg, profile,
                                    exchanges, entry["probe"])
        if serr:
            errors.append(f"student(turn{t}): {serr}")
        exchanges.append({"tutor": entry["probe"], "student": reply, "source": "probe"})

    return {
        "task_id": row.get("task_id"),
        "difficulty": row.get("difficulty"),
        "profile_id": prof["profile_id"],
        "archetype": prof["archetype"],
        "condition": condition,
        "budget": budget,
        "honor_stop": honor_stop,
        "p_true": profile,
        "p_pred": last_belief,
        "turns": turn_log,
        "exchanges": exchanges,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tutor-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--tutor-model", required=True)
    ap.add_argument("--tutor-api-key", default="not-needed")
    ap.add_argument("--tutor-temperature", type=float, default=0.3)
    ap.add_argument("--student-base-url", default=None,
                    help="defaults to --tutor-base-url (same server, different model)")
    ap.add_argument("--student-model", required=True)
    ap.add_argument("--student-api-key", default="not-needed")
    ap.add_argument("--student-temperature", type=float, default=0.7,
                    help="students are noisy; keep this above the tutor's")
    ap.add_argument("--max-tokens", type=int, default=1536)
    # Per-role, because this is a vendor-specific knob: sending it to a model
    # that doesn't implement it makes the server reject the whole request.
    ap.add_argument("--tutor-reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--student-reasoning-effort", default=None, choices=["low", "medium", "high"])

    ap.add_argument("--turns", type=int, default=6, help="probe budget N")
    ap.add_argument("--problems", type=int, default=3,
                    help="how many problems each profile is interviewed on")
    ap.add_argument("--difficulty", default="Easy",
                    help="dataset difficulty filter; 'any' to disable (CS1 => Easy)")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n-profiles", type=int, default=None,
                    help="sample exactly this many profiles across the archetype mix "
                         "(overrides --profiles-per-archetype/--archetypes)")
    ap.add_argument("--profiles-per-archetype", type=int, default=1)
    ap.add_argument("--archetypes", nargs="*", default=ARCHETYPES, choices=ARCHETYPES)
    ap.add_argument("--conditions", nargs="*", default=["interactive"], choices=CONDITIONS)
    ap.add_argument("--ignore-stop", action="store_true",
                    help="keep probing to the full budget even after the tutor declares "
                         "it is done (measures what early stopping cost)")
    ap.add_argument("--profile-seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", default="rollouts.jsonl")
    args = ap.parse_args()

    ds = load_split(args.split, None)
    rows = [r for r in ds
            if args.difficulty == "any" or r.get("difficulty") == args.difficulty]
    rows = rows[:args.problems]
    if not rows:
        sys.exit(f"no problems matched difficulty={args.difficulty!r}")

    profs = (sample_n_profiles(args.n_profiles, args.profile_seed) if args.n_profiles
             else sample_profiles(args.profiles_per_archetype, args.profile_seed,
                                  args.archetypes))

    def _effort(v):
        return {"reasoning_effort": v} if v else {}

    tutor_cfg = ModelConfig(base_url=args.tutor_base_url, model=args.tutor_model,
                            api_key=args.tutor_api_key, temperature=args.tutor_temperature,
                            max_tokens=args.max_tokens,
                            extra_body=_effort(args.tutor_reasoning_effort))
    student_cfg = ModelConfig(base_url=args.student_base_url or args.tutor_base_url,
                              model=args.student_model, api_key=args.student_api_key,
                              temperature=args.student_temperature,
                              max_tokens=args.max_tokens,
                              extra_body=_effort(args.student_reasoning_effort))
    tutor_client, student_client = make_client(tutor_cfg), make_client(student_cfg)

    jobs = [(row, prof, cond)
            for cond in args.conditions for prof in profs for row in rows]
    print(f"{len(jobs)} episodes = {len(args.conditions)} conditions x "
          f"{len(profs)} profiles x {len(rows)} problems", file=sys.stderr)

    meta = {
        "_meta": True,
        "tutor_model": args.tutor_model,
        "student_model": args.student_model,
        "turns": args.turns,
        "conditions": args.conditions,
        "profile_seed": args.profile_seed,
        "stages": STAGE_IDS,
    }

    with open(args.output, "w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        out_f.write(json.dumps(meta) + "\n")
        futures = [pool.submit(run_episode, tutor_client, tutor_cfg, student_client,
                               student_cfg, row, prof, args.turns, cond,
                               not args.ignore_stop)
                   for row, prof, cond in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="diagnosis"):
            out_f.write(json.dumps(fut.result()) + "\n")
            out_f.flush()

    print(f"Wrote {len(jobs)} episodes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
