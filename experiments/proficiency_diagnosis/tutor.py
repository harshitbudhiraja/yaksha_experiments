"""The tutor under test: given a problem and a transcript with a student, it
must (a) choose the next diagnostic probe and (b) report its current belief
over the student's 12-stage profile, every single turn.

Requiring a full belief vector at *every* turn (not just at the end) is what
makes convergence cost measurable -- it gives an error-vs-turn curve rather
than a single endpoint. Requiring an explicit `stop` flag separates "the tutor
thinks it is done" from "the tutor ran out of budget", which are different
diagnoses of the same final number.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.model_client import ModelConfig, chat_messages
from common.stages import LEVELS, STAGE_IDS, rubric_for_tutor

TUTOR_SYSTEM = """\
You are an expert CS1 instructor running a diagnostic interview with one student \
about one problem. Your only goal is to work out *where in their problem-solving \
process* they are weak -- not to teach, not to give them the answer.

{rubric}

You get at most {budget} turns. On each turn you send the student one message and \
see their reply. After each reply you must report your current best estimate of \
their level (1-5) for all 12 stages, even the ones you have no evidence about yet \
(guess, and refine later).

Probing advice: a probe that only asks "do you understand?" tells you nothing. \
Good probes force the student to *perform* the stage you are unsure about -- ask \
them to trace an input, predict an output, justify a choice, produce a test case, \
find a bug you point at, or state a complexity. You may only send messages a real \
tutor could send; do not ask the student to reveal hidden state or rate themselves \
numerically.

Respond on EVERY turn with a single JSON object and nothing else:

{{
  "reasoning": "<2-4 sentences: what the last reply told you and what you still need>",
  "target_stages": ["S3", "S5"],
  "probe": "<the exact message to send the student>",
  "belief": {{"S1": 3, "S2": 3, "S3": 3, "S4": 3, "S5": 3, "S6": 3,
              "S7": 3, "S8": 3, "S9": 3, "S10": 3, "S11": 3, "S12": 3}},
  "confidence": {{"S1": 0.2, "S2": 0.2, "S3": 0.2, "S4": 0.2, "S5": 0.2, "S6": 0.2,
                  "S7": 0.2, "S8": 0.2, "S9": 0.2, "S10": 0.2, "S11": 0.2, "S12": 0.2}},
  "stop": false
}}

`belief` must contain all 12 stages with integer values 1-5. `confidence` is 0.0-1.0 \
per stage. Set "stop": true only when further probing would not change your profile; \
your belief on that turn becomes final and the interview ends early. Use the full \
1-5 range -- a student who cannot do a stage at all is a 1, and refusing to assign \
extreme values will make your diagnosis wrong.
"""

FINAL_TURN_NOTE = """\
This is your LAST turn -- the interview is over and there will be no further student \
reply. Output the same JSON object with your FINAL belief, "stop": true, and "probe": "".
"""

BLIND_INSTRUCTION = """\
You will not get to interact with this student at all. Based only on the problem they \
were assigned and general knowledge of CS1 students, report your best-guess profile. \
Output the JSON object with "stop": true and "probe": "".
"""

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _coerce_level(v: Any) -> Optional[int]:
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return iv if iv in LEVELS else max(1, min(5, iv))


def parse_tutor_output(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Tolerant extraction of the tutor's JSON action. Returns (action, error).

    Format failures are data, not crashes: they are reported as a compliance
    metric, and the caller carries the previous belief forward so one bad turn
    doesn't void an episode.
    """
    candidates: List[str] = []
    block = JSON_BLOCK_RE.search(text)
    if block:
        candidates.append(block.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append(text)

    obj = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return None, "no parseable JSON object in tutor response"

    raw_belief = obj.get("belief")
    if not isinstance(raw_belief, dict):
        return None, "tutor response has no 'belief' object"

    belief, missing = {}, []
    for sid in STAGE_IDS:
        lvl = _coerce_level(raw_belief.get(sid))
        if lvl is None:
            missing.append(sid)
        else:
            belief[sid] = lvl
    if missing:
        return None, f"belief missing/invalid for stages: {missing}"

    raw_conf = obj.get("confidence") if isinstance(obj.get("confidence"), dict) else {}
    confidence = {}
    for sid in STAGE_IDS:
        try:
            confidence[sid] = max(0.0, min(1.0, float(raw_conf.get(sid))))
        except (TypeError, ValueError):
            confidence[sid] = None

    targets = obj.get("target_stages")
    targets = [t for t in targets if t in STAGE_IDS] if isinstance(targets, list) else []

    return {
        "reasoning": str(obj.get("reasoning") or "")[:2000],
        "target_stages": targets,
        "probe": str(obj.get("probe") or ""),
        "belief": belief,
        "confidence": confidence,
        "stop": bool(obj.get("stop")),
    }, None


RETRY_NOTE = """\
Your previous response could not be parsed ({err}). Respond with ONLY the JSON \
object described above -- no prose before or after it, all 12 stages present in \
"belief" with integer values 1-5.
"""


def tutor_turn(client, cfg: ModelConfig, problem_text: str,
               exchanges: List[Dict[str, str]], turn: int, budget: int,
               is_final: bool, blind: bool = False,
               retries: int = 1) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """One tutor turn. Returns (action | None, raw_text, error).

    A malformed response is retried once with the parser's complaint attached;
    format non-compliance is still recorded (evaluate.py reports it) but one
    bad turn no longer voids the episode."""
    transcript = "\n\n".join(
        f"--- turn {i} ---\nYOU: {ex['tutor']}\nSTUDENT: {ex['student']}"
        for i, ex in enumerate(exchanges)
    ) or "(no exchanges yet)"

    user = (
        f"Problem the student was assigned:\n\n{problem_text}\n\n"
        f"Interview transcript so far:\n\n{transcript}\n\n"
        f"This is turn {turn} of at most {budget}."
    )
    if blind:
        user += "\n\n" + BLIND_INSTRUCTION
    elif is_final:
        user += "\n\n" + FINAL_TURN_NOTE

    messages = [
        {"role": "system", "content": TUTOR_SYSTEM.format(
            rubric=rubric_for_tutor(), budget=budget)},
        {"role": "user", "content": user},
    ]
    for attempt in range(retries + 1):
        raw, err = chat_messages(client, cfg, messages)
        if err:
            return None, raw, err
        action, perr = parse_tutor_output(raw)
        if action is not None or attempt == retries:
            return action, raw, perr
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": RETRY_NOTE.format(err=perr)},
        ]
