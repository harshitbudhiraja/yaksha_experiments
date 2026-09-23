#!/usr/bin/env python3
"""Scores the rollouts from run_rollouts.py.

Headline metrics (the two the design asks for):

  localization_mae   mean |P_pred(c) - P_true(c)| over the 12 stages
  convergence_cost   turns spent before the diagnosis stopped moving

`localization_mae` on its own is not interpretable, because a tutor that
answers "3 everywhere" already scores ~0.9 against a realistic profile
distribution. So every run is reported against three references:

  best_constant_mae   the best possible single-number-everywhere answer,
                      chosen with hindsight over this exact profile set
  random_mae          uniform random levels
  skill_over_constant best_constant_mae - localization_mae  (>0 means the
                      tutor extracted real information; <=0 means it did not)

Convergence is reported three ways, because "when did it converge" is
ambiguous and the three disagree in informative ways:
  declared_stop_turn  when the tutor said it was done (may be overconfident)
  stabilization_turn  when the belief vector stopped changing (eps/patience)
  turns_to_best       when the error curve first reached its final value

Usage:
    python evaluate.py --rollouts rollouts.jsonl \
        --output episode_scores.jsonl --summary summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.stages import CODE_LEVEL_IDS, LEVELS, PRE_CODING_IDS, STAGE_IDS

# Student utterances that would hand the tutor the answer instead of making it
# diagnose. Any hit means that episode's difficulty was artificially lowered.
LEAK_PATTERNS = [
    re.compile(r"\bS(?:[1-9]|1[0-2])\b"),
    re.compile(r"\bproficiency\b", re.I),
    re.compile(r"\b(?:level|rated?|score)\s*(?:of\s*)?[1-5]\s*(?:/|out of)\s*5", re.I),
    re.compile(r"\b[1-5]\s*(?:/|out of)\s*5\b"),
    re.compile(r"\b(?:my\s+)?(?:hidden\s+)?(?:skill\s+)?profile\b", re.I),
    re.compile(r"\brubric\b", re.I),
    re.compile(r"\bsimulat", re.I),
]


def mae(pred: Dict[str, int], true: Dict[str, int], stages: List[str] = None) -> float:
    stages = stages or STAGE_IDS
    return sum(abs(pred[s] - true[s]) for s in stages) / len(stages)


def spearman(pred: Dict[str, int], true: Dict[str, int]) -> Optional[float]:
    """Rank correlation between the predicted and true profile *shape*. A tutor
    can be systematically 1 point too generous (bad MAE) and still rank the
    student's stages perfectly (good rho) -- that is a calibration problem, not
    a diagnosis problem, and the two need separating."""
    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    a, b = ranks([pred[s] for s in STAGE_IDS]), ranks([true[s] for s in STAGE_IDS])
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return None if den == 0 else num / den


def weak_recall(pred: Dict[str, int], true: Dict[str, int], k: int = 3) -> float:
    """Of the student's k genuinely weakest stages, how many are in the tutor's
    predicted bottom-k? This is the metric that matches what a tutor would
    actually *do* next -- pick what to remediate. Ties in the true profile are
    resolved generously (any stage tied at the k-th weakest level counts)."""
    true_sorted = sorted(STAGE_IDS, key=lambda s: true[s])
    cutoff = true[true_sorted[k - 1]]
    true_weak = {s for s in STAGE_IDS if true[s] <= cutoff}
    pred_weak = set(sorted(STAGE_IDS, key=lambda s: pred[s])[:k])
    hits = len(pred_weak & true_weak)
    return hits / min(k, len(true_weak))


def belief_curve(ep: Dict[str, Any]) -> List[Dict[str, int]]:
    """Belief after each turn, carried forward to the full budget so error
    curves from early-stopping and full-budget episodes are comparable."""
    curve = [t["belief"] for t in ep["turns"] if t["belief"]]
    if not curve:
        return []
    while len(curve) < ep["budget"]:
        curve.append(curve[-1])
    return curve


def stabilization_turn(curve: List[Dict[str, int]], eps: float = 0.25,
                       patience: int = 2) -> Optional[int]:
    """First turn after which the belief vector moves by <= eps (mean abs change
    per stage) for `patience` consecutive turns."""
    if len(curve) < patience + 1:
        return None
    for i in range(len(curve) - patience):
        if all(mae(curve[i + j], curve[i + j + 1]) <= eps for j in range(patience)):
            return i + 1
    return None


def score_episode(ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    true, pred = ep["p_true"], ep["p_pred"]
    if not pred:
        return None

    curve = belief_curve(ep)
    curve_mae = [mae(b, true) for b in curve]
    final_mae = curve_mae[-1] if curve_mae else mae(pred, true)

    declared = next((t["turn"] for t in ep["turns"] if t["declared_stop"]), None)
    turns_to_best = next((i + 1 for i, m in enumerate(curve_mae)
                          if abs(m - curve_mae[-1]) < 1e-9), None)

    student_text = " ".join(x["student"] or "" for x in ep.get("exchanges", []))
    leaks = [p.pattern for p in LEAK_PATTERNS if p.search(student_text)]

    probed = Counter(s for t in ep["turns"] for s in t["target_stages"])

    return {
        "task_id": ep["task_id"],
        "profile_id": ep["profile_id"],
        "archetype": ep["archetype"],
        "condition": ep["condition"],

        "localization_mae": final_mae,
        "mae_pre_coding": mae(pred, true, PRE_CODING_IDS),
        "mae_code_level": mae(pred, true, CODE_LEVEL_IDS),
        "exact_match_rate": sum(pred[s] == true[s] for s in STAGE_IDS) / len(STAGE_IDS),
        "within_one_rate": sum(abs(pred[s] - true[s]) <= 1 for s in STAGE_IDS) / len(STAGE_IDS),
        "signed_bias": sum(pred[s] - true[s] for s in STAGE_IDS) / len(STAGE_IDS),
        "spearman_rho": spearman(pred, true),
        "weak_recall_at3": weak_recall(pred, true, 3),
        "pred_level_spread": (max(pred.values()) - min(pred.values())),

        "turns_used": len(ep["turns"]),
        "declared_stop_turn": declared,
        "stabilization_turn": stabilization_turn(curve),
        "turns_to_best": turns_to_best,
        "auc_mae": statistics.fmean(curve_mae) if curve_mae else None,
        "mae_by_turn": curve_mae,

        "parse_fail_rate": sum(not t["parse_ok"] for t in ep["turns"]) / max(1, len(ep["turns"])),
        "student_leak_hits": leaks,
        "probed_stages": dict(probed),
        "per_stage_abs_err": {s: abs(pred[s] - true[s]) for s in STAGE_IDS},
        "errors": ep.get("errors", []),
    }


def baselines(trues: List[Dict[str, int]]) -> Dict[str, Any]:
    """Reference points computed on the same profile population the tutor saw."""
    const = {c: statistics.fmean(mae({s: c for s in STAGE_IDS}, t) for t in trues)
             for c in LEVELS}
    best_c = min(const, key=const.get)
    rng_mae = statistics.fmean(
        statistics.fmean(abs(c - t[s]) for s in STAGE_IDS for c in LEVELS) for t in trues)
    return {
        "constant_mae_by_level": const,
        "best_constant_level": best_c,
        "best_constant_mae": const[best_c],
        "random_uniform_mae": rng_mae,
    }


def _agg(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    def m(key):
        vals = [s[key] for s in scores if s.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    out = {k: m(k) for k in [
        "localization_mae", "mae_pre_coding", "mae_code_level", "exact_match_rate",
        "within_one_rate", "signed_bias", "spearman_rho", "weak_recall_at3",
        "pred_level_spread", "turns_used", "declared_stop_turn", "stabilization_turn",
        "turns_to_best", "auc_mae", "parse_fail_rate"]}
    out["n_episodes"] = len(scores)
    maes = [s["localization_mae"] for s in scores]
    out["localization_mae_stdev"] = statistics.stdev(maes) if len(maes) > 1 else 0.0
    out["student_leak_rate"] = statistics.fmean(
        [1.0 if s["student_leak_hits"] else 0.0 for s in scores]) if scores else None
    out["declared_stop_rate"] = statistics.fmean(
        [1.0 if s["declared_stop_turn"] else 0.0 for s in scores]) if scores else None

    per_stage = defaultdict(list)
    for s in scores:
        for stage, err in s["per_stage_abs_err"].items():
            per_stage[stage].append(err)
    out["per_stage_mae"] = {k: statistics.fmean(v) for k, v in sorted(per_stage.items())}

    probe_counts = Counter()
    for s in scores:
        probe_counts.update(s["probed_stages"])
    out["probe_targeting_counts"] = dict(sorted(probe_counts.items()))
    unprobed = [s for s in STAGE_IDS if s not in probe_counts]
    out["never_probed_stages"] = unprobed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--output", default="episode_scores.jsonl")
    ap.add_argument("--summary", default="summary.json")
    args = ap.parse_args()

    meta, episodes = {}, []
    with open(args.rollouts) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            (meta.update(rec) if rec.get("_meta") else episodes.append(rec))

    scores, dropped = [], 0
    for ep in episodes:
        s = score_episode(ep)
        if s is None:
            dropped += 1
        else:
            scores.append(s)

    with open(args.output, "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    summary: Dict[str, Any] = {
        "tutor_model": meta.get("tutor_model"),
        "student_model": meta.get("student_model"),
        "n_episodes_scored": len(scores),
        "n_episodes_dropped": dropped,
        "baselines": baselines([ep["p_true"] for ep in episodes]),
        "by_condition": {},
        "by_archetype": {},
    }

    bc = summary["baselines"]["best_constant_mae"]
    for cond in sorted({s["condition"] for s in scores}):
        sub = [s for s in scores if s["condition"] == cond]
        agg = _agg(sub)
        agg["skill_over_best_constant"] = bc - agg["localization_mae"]
        summary["by_condition"][cond] = agg

    for arch in sorted({s["archetype"] for s in scores}):
        sub = [s for s in scores if s["archetype"] == arch and s["condition"] == "interactive"]
        if sub:
            summary["by_archetype"][arch] = _agg(sub)

    inter = summary["by_condition"].get("interactive")
    one = summary["by_condition"].get("one_shot")
    blind = summary["by_condition"].get("blind")
    if inter and one:
        summary["value_of_probing"] = one["localization_mae"] - inter["localization_mae"]
    if inter and blind:
        summary["value_of_seeing_student"] = blind["localization_mae"] - inter["localization_mae"]

    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "by_archetype"}, indent=2))
    print(f"\nWrote {len(scores)} episode scores to {args.output} "
          f"and summary to {args.summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
