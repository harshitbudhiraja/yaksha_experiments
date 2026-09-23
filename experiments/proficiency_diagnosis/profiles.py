"""Ground-truth proficiency profiles P_true fed to the simulated student.

Uniform-random profiles alone are a weak test set: they average out, and a
tutor that always answers "3 everywhere" scores well against them. The
archetypes below exist so the evaluation can separate a tutor that is
*measuring* from one that is *regressing to the mean*:

  - `spiky_deficit`  one stage at 1, everything else at 4. The sharpest
                     localization test -- can the tutor find the single hole?
  - `band_split`     pre-coding strong / code-level weak, or the reverse.
                     Tests whether the tutor distinguishes "can't plan" from
                     "can't code", which is the distinction that changes what
                     you'd actually teach next.
  - `flat_mid`       3 everywhere. A trap for tutors with a high-score bias;
                     also the profile the constant-3 baseline gets for free.
  - `novice` / `strong`  the ends of the range, where score compression
                     (tutors refusing to use 1 or 5) shows up.
  - `uniform_random` unstructured control.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.stages import CODE_LEVEL_IDS, PRE_CODING_IDS, STAGE_IDS, Profile

ARCHETYPES = [
    "spiky_deficit",
    "band_split_plan_weak",
    "band_split_code_weak",
    "flat_mid",
    "novice",
    "strong",
    "uniform_random",
]


def _jitter(base: int, rng: random.Random, p: float = 0.35) -> int:
    """Nudge a level by +/-1 with probability p, so archetypes aren't
    degenerate constants the tutor could luck into."""
    if rng.random() >= p:
        return base
    return max(1, min(5, base + rng.choice([-1, 1])))


def make_profile(archetype: str, rng: random.Random,
                 weak_stage: Optional[str] = None) -> Profile:
    if archetype == "spiky_deficit":
        # `weak_stage` lets the caller cycle the deficit across all 12 stages
        # instead of sampling it, so per-stage coverage is even.
        weak = weak_stage or rng.choice(STAGE_IDS)
        return {s: (1 if s == weak else _jitter(4, rng)) for s in STAGE_IDS}

    if archetype == "band_split_plan_weak":
        return {**{s: _jitter(2, rng) for s in PRE_CODING_IDS},
                **{s: _jitter(4, rng) for s in CODE_LEVEL_IDS}}

    if archetype == "band_split_code_weak":
        return {**{s: _jitter(4, rng) for s in PRE_CODING_IDS},
                **{s: _jitter(2, rng) for s in CODE_LEVEL_IDS}}

    if archetype == "flat_mid":
        return {s: 3 for s in STAGE_IDS}

    if archetype == "novice":
        return {s: _jitter(2, rng) for s in STAGE_IDS}

    if archetype == "strong":
        return {s: _jitter(4, rng) for s in STAGE_IDS}

    if archetype == "uniform_random":
        return {s: rng.randint(1, 5) for s in STAGE_IDS}

    raise ValueError(f"unknown archetype: {archetype}")


# Mix used when sampling a fixed-size population. Weights are relative, and
# chosen so that: unstructured profiles dominate (broad coverage of the
# s1..s12 space), spiky deficits get ~2 per stage (even per-stage coverage of
# the sharpest localization test), and flat_mid keeps a handful of *identical*
# profiles on purpose -- repeated draws of the same student give a free
# test-retest reliability read on the tutor.
DEFAULT_MIX: Dict[str, int] = {
    "uniform_random": 30,
    "spiky_deficit": 24,
    "band_split_plan_weak": 12,
    "band_split_code_weak": 12,
    "novice": 10,
    "strong": 8,
    "flat_mid": 4,
}


def sample_n_profiles(n: int = 100, seed: int = 0,
                      mix: Optional[Dict[str, int]] = None) -> List[Dict]:
    """Exactly `n` ground-truth profiles, apportioned across archetypes by
    `mix` (largest-remainder rounding). Deterministic given (n, seed, mix)."""
    mix = mix or DEFAULT_MIX
    total = sum(mix.values())
    exact = {a: n * w / total for a, w in mix.items()}
    counts = {a: int(v) for a, v in exact.items()}
    for a in sorted(mix, key=lambda a: exact[a] - counts[a], reverse=True):
        if sum(counts.values()) >= n:
            break
        counts[a] += 1

    rng = random.Random(seed)
    out, spiky_i = [], 0
    for arch in mix:  # insertion order -> stable ids across runs
        for i in range(counts[arch]):
            weak = None
            if arch == "spiky_deficit":
                weak = STAGE_IDS[spiky_i % len(STAGE_IDS)]
                spiky_i += 1
            out.append({
                "profile_id": f"{arch}:{i}",
                "archetype": arch,
                "weak_stage": weak,
                "profile": make_profile(arch, rng, weak_stage=weak),
            })
    return out


def sample_profiles(n_per_archetype: int = 1, seed: int = 0,
                    archetypes: Optional[List[str]] = None) -> List[Dict]:
    """Deterministic given (seed, archetypes, n_per_archetype) so two models can
    be compared on exactly the same student population."""
    archetypes = archetypes or ARCHETYPES
    rng = random.Random(seed)
    out = []
    for arch in archetypes:
        for i in range(n_per_archetype):
            out.append({
                "profile_id": f"{arch}:{i}",
                "archetype": arch,
                "profile": make_profile(arch, rng),
            })
    return out
