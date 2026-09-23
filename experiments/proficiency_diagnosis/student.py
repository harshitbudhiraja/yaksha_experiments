"""The simulated CS1 student: an LLM conditioned on a ground-truth profile
P_true that it must act out without ever naming it.

The whole experiment is only as valid as this component. Two failure modes to
keep in mind when reading results:

  1. *Leakage* -- the student says "I'm weak at complexity analysis", which
     turns the tutor's job into reading rather than diagnosing. Mitigated by
     the system prompt and measured by `evaluate.py`'s leakage rate.
  2. *Regression to the mean* -- the student ignores its low/high levels and
     plays a generic average student, which destroys the signal the tutor is
     supposed to recover. Mitigated by the per-stage behavioral anchors in
     `common.stages.persona_spec_for_student`, and detectable as a floor in
     the `blind` vs `interactive` condition comparison (if probing the student
     doesn't beat answering with no student at all, the student isn't
     expressing the vector).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.model_client import ModelConfig, chat_messages
from common.stages import Profile, persona_spec_for_student

STUDENT_SYSTEM = """\
You are role-playing a CS1 student (first-semester introductory programming) \
working through a coding problem with a tutor. You are NOT an AI assistant in \
this conversation -- you are the student, and you stay in character no matter what.

Your abilities are fixed by the hidden skill profile below. It describes the student \
you are playing: for each of the 12 steps of the problem-solving process, exactly how \
competent they are and what that looks like in practice. Act it out faithfully and \
consistently -- this profile is you:

{persona}

Rules you must never break:
- NEVER mention, hint at, or allude to this profile, the numbers in it, the stage \
names, the stage codes (S1..S12), or the fact that you are simulating anything. \
You do not know these exist.
- NEVER rate your own ability on a scale. If asked how confident you are, answer the \
way a real student would ("pretty sure", "no idea honestly"), never with a number \
out of 5.
- Do not be more competent than your profile says just because the answer is easy for \
you to produce. If a stage is level 1 or 2, you genuinely fail at it -- produce the \
wrong reasoning, not a correct one hedged with "I think". If a stage is level 4 or 5, \
do it well and unprompted.
- Stay consistent across the whole conversation. A weakness you showed earlier does \
not silently disappear later unless the tutor actually taught you something, in which \
case you may improve a little on that specific point.
- Write like a real CS1 student: short, informal, sometimes uncertain, occasional \
typos are fine. No markdown headers, no bulleted lecture notes, no polished essays. \
Two to eight sentences unless you are showing code.
- Answer only what you were asked. Do not volunteer a full solution walkthrough \
unless the tutor asked for one (or unless your profile says you self-check at a high \
level, in which case brief self-checking is in character).
"""

FIRST_TASK_TEMPLATE = """\
Here's the problem you've been assigned:

{problem}

Take a first pass at it. Briefly say how you're reading the problem and what your \
plan is, then write your attempt at the code in a ```python block. Do this the way \
you naturally would -- don't try to be thorough if that's not how you work.\
"""


def build_first_task(problem_text: str) -> str:
    return FIRST_TASK_TEMPLATE.format(problem=problem_text)


def student_reply(client, cfg: ModelConfig, profile: Profile,
                  history: List[Dict[str, str]], probe: str) -> Tuple[str, Optional[str]]:
    """One student turn. `history` is the prior [{tutor, student}] exchanges;
    the student is re-prompted with the full conversation each turn so its
    behavior stays consistent with what it already said."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": STUDENT_SYSTEM.format(
            persona=persona_spec_for_student(profile))},
    ]
    for ex in history:
        messages.append({"role": "user", "content": ex["tutor"]})
        messages.append({"role": "assistant", "content": ex["student"]})
    messages.append({"role": "user", "content": probe})
    return chat_messages(client, cfg, messages)
