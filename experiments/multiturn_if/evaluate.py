#!/usr/bin/env python3
"""Score run_dialogue.py's logs with the CodeIF-Bench metric family.

All four metrics are derived from the per-turn status matrix the dialogue log
already contains -- no model is re-run here, so re-scoring is free.

Let S_t be the instructions issued up to and including turn t (the base prompt
counts as one), i_t the instruction that turn t was addressing, and P_t the
instructions issued before t that the model had satisfied at some earlier turn.

  IA_t  = 1 if every unit test of i_t passes on turn t's code
          -- "did it do what I just asked"
  CA_t  = |{n in S_t : n passes at t}| / |S_t|
          -- "how much of the conversation is still satisfied"
  IFR_t = |{n in P_t : n fails at t}| / |P_t|
          -- "of what it had already got right, how much did it break"
  CIF   = |{n passing on the final code}| over ALL of the task's instructions
          (per dialogue) -- "how many instructions survived to the end"

CIF counts every instruction the task defines, not just the ones a mode got
around to issuing, so it is comparable across modes: `dynamic` only asks about
instructions the current code fails, so an issued-only count would score it
against a harder, self-selected subset than `static`. `CIF_issued` keeps the
issued-only variant.

Two things this reports that the upstream scripts don't:

  * `forgetting_by_age` -- IFR split by how many turns ago the forgotten
    instruction was issued. This is the context-management curve; a flat one
    means the loss is not about distance in the context window.
  * `unprompted_pass_rate` -- how many instructions already pass on turn 1,
    before being asked for. Some requirements (e.g. a complexity bound) are
    satisfied by any reasonable solution, which inflates CA; subtract this
    baseline before reading CA as instruction following.

`upstream_ife` reproduces the reference implementation's `IFE = CA_t / t` for
comparability with the numbers in the paper's repo. It mechanically decays with
turn index (it divides a ratio by a turn count), so prefer CIF and final CA.

Usage:
    python evaluate.py --dialogues mtif_dynamic.jsonl \
        --output mtif_results.jsonl --summary mtif_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

BASE_INSTRUCTION = "Base"


def _mean(xs: List[float]) -> float:
    return mean(xs) if xs else 0.0


def score_dialogue(log: Dict[str, Any]) -> Dict[str, Any]:
    turns = log["turns"]
    satisfied_at: Dict[str, int] = {}   # instruction -> first turn it passed on
    per_turn: List[Dict[str, Any]] = []
    forget_events: List[Dict[str, Any]] = []

    for turn in turns:
        t = turn["turn"]
        status = turn["status"]
        issued = turn["issued_so_far"]
        name = turn["instruction_name"]

        prev_satisfied = [n for n in issued if n in satisfied_at and satisfied_at[n] < t]
        broken = [n for n in prev_satisfied if not status.get(n)]
        for n in broken:
            forget_events.append({"instruction": n, "age": t - satisfied_at[n], "broken": True})
        for n in prev_satisfied:
            if n not in broken:
                forget_events.append({"instruction": n, "age": t - satisfied_at[n], "broken": False})

        per_turn.append({
            "turn": t,
            "instruction_name": name,
            "is_retry": turn["is_retry"],
            "IA": 1.0 if status.get(name) else 0.0,
            "CA": _mean([1.0 if status.get(n) else 0.0 for n in issued]),
            "IFR": (len(broken) / len(prev_satisfied)) if prev_satisfied else 0.0,
            "n_issued": len(issued),
            "broken": broken,
        })

        for n in issued:
            if status.get(n) and n not in satisfied_at:
                satisfied_at[n] = t

    final_status = turns[-1]["status"] if turns else {}
    final_issued = turns[-1]["issued_so_far"] if turns else []
    all_names = log["instruction_order"]

    # Instructions passing on turn 1 that had not been asked for yet.
    unprompted = []
    if turns:
        first = turns[0]
        unprompted = [n for n in all_names
                      if n not in first["issued_so_far"] and first["status"].get(n)]

    return {
        "task_id": log["task_id"],
        "mode": log["mode"],
        "n_turns": len(turns),
        "n_instructions": len(all_names),
        "per_turn": per_turn,
        "forget_events": forget_events,
        "CIF": sum(1 for n in all_names if final_status.get(n)),
        "CIF_excl_base": sum(1 for n in all_names if n != BASE_INSTRUCTION and final_status.get(n)),
        "CIF_issued": sum(1 for n in final_issued if final_status.get(n)),
        "n_issued": len(final_issued),
        "final_CA": _mean([1.0 if final_status.get(n) else 0.0 for n in final_issued]),
        "base_passes_at_end": bool(final_status.get(BASE_INSTRUCTION)),
        "n_unprompted_pass": len(unprompted),
        "n_unissued_turn1": len([n for n in all_names if turns and n not in turns[0]["issued_so_far"]]),
        "first_attempt_IA": {
            pt["instruction_name"]: pt["IA"]
            for pt in per_turn if not pt["is_retry"] and pt["instruction_name"]
        },
        "retry_IA": {
            pt["instruction_name"]: pt["IA"]
            for pt in per_turn if pt["is_retry"] and pt["instruction_name"]
        },
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_turns = [pt for r in results for pt in r["per_turn"]]
    events = [e for r in results for e in r["forget_events"]]

    by_type_first = defaultdict(list)
    by_type_retry = defaultdict(list)
    for r in results:
        for name, ia in r["first_attempt_IA"].items():
            by_type_first[name].append(ia)
        for name, ia in r["retry_IA"].items():
            by_type_retry[name].append(ia)

    by_turn = defaultdict(list)
    for pt in all_turns:
        by_turn[pt["turn"]].append(pt)

    by_age = defaultdict(list)
    for e in events:
        by_age[e["age"]].append(1.0 if e["broken"] else 0.0)

    forgotten_by_type = defaultdict(list)
    for e in events:
        forgotten_by_type[e["instruction"]].append(1.0 if e["broken"] else 0.0)

    return {
        "n_dialogues": len(results),
        "mode": results[0]["mode"] if results else None,
        "IA": _mean([pt["IA"] for pt in all_turns]),
        "IA_first_attempt": _mean([pt["IA"] for pt in all_turns if not pt["is_retry"]]),
        "CA": _mean([pt["CA"] for pt in all_turns]),
        "IFR": _mean([pt["IFR"] for pt in all_turns]),
        "CIF": _mean([float(r["CIF"]) for r in results]),
        "CIF_excl_base": _mean([float(r["CIF_excl_base"]) for r in results]),
        "CIF_issued": _mean([float(r["CIF_issued"]) for r in results]),
        "mean_instructions_issued": _mean([float(r["n_issued"]) for r in results]),
        "final_CA": _mean([r["final_CA"] for r in results]),
        "base_retained_rate": _mean([1.0 if r["base_passes_at_end"] else 0.0 for r in results]),
        "mean_turns": _mean([float(r["n_turns"]) for r in results]),
        "unprompted_pass_rate": (
            sum(r["n_unprompted_pass"] for r in results)
            / sum(r["n_unissued_turn1"] for r in results)
            if sum(r["n_unissued_turn1"] for r in results) else 0.0
        ),
        "by_instruction_type": {
            name: {
                "n": len(ias),
                "IA_first_attempt": _mean(ias),
                "IA_after_feedback": (_mean(by_type_retry[name]) if by_type_retry.get(name) else None),
                "n_retries": len(by_type_retry.get(name, [])),
                "forget_rate_once_satisfied": _mean(forgotten_by_type.get(name, [])),
            }
            for name, ias in sorted(by_type_first.items())
        },
        "by_turn": {
            str(t): {
                "n": len(pts),
                "IA": _mean([p["IA"] for p in pts]),
                "CA": _mean([p["CA"] for p in pts]),
                "IFR": _mean([p["IFR"] for p in pts]),
                "upstream_ife": _mean([p["CA"] / t for p in pts]),
            }
            for t, pts in sorted(by_turn.items())
        },
        "forgetting_by_age": {
            str(age): {"n": len(xs), "forget_rate": _mean(xs)}
            for age, xs in sorted(by_age.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--output", default="mtif_results.jsonl")
    ap.add_argument("--summary", default="mtif_summary.json")
    args = ap.parse_args()

    logs = []
    with open(args.dialogues) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    results = [score_dialogue(log) for log in logs if log["turns"]]

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = summarize(results)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
