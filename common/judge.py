"""Optional LLM-judge for calls the diff-pattern heuristics in bug_taxonomy.py
can't make reliably (e.g. "is this plausibly a mistake a person would make?",
or disambiguating 'logic' vs 'semantic_algorithmic'). Opt-in via --use-llm-judge
in each experiment's evaluate.py -- every metric that depends on this should
also be computable, more cheaply and less accurately, without it.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from common.bug_taxonomy import BUG_TYPES
from common.model_client import ModelConfig

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_TAXONOMY_LIST = "\n".join(f"- {b.id}: {b.description}" for b in BUG_TYPES)


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def classify_bug_llm(
    client: OpenAI, cfg: ModelConfig, reference: str, buggy: str
) -> Dict[str, Any]:
    """Ask a judge model to classify the bug type and rate plausibility.
    Returns {"bug_type": str|None, "plausibility": float|None, "raw": str,
    "parse_error": bool}.
    """
    prompt = (
        "You are auditing a bug injected into a correct solution for a coding "
        "tutoring tool. Given the reference (correct) solution and the mutated "
        "(buggy) solution, classify the bug and rate how plausible it is that a "
        "student would make this mistake themselves.\n\n"
        f"Bug taxonomy:\n{_TAXONOMY_LIST}\n\n"
        f"### Reference (correct) solution:\n```python\n{reference}\n```\n\n"
        f"### Buggy solution:\n```python\n{buggy}\n```\n\n"
        "Respond with ONLY a JSON object: "
        '{"bug_type": "<one taxonomy id>", "plausibility": <0.0-1.0>, '
        '"rationale": "<one sentence>"}'
    )
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content or ""
    parsed = _parse_json_response(raw)
    if parsed is None:
        return {"bug_type": None, "plausibility": None, "raw": raw, "parse_error": True}
    return {
        "bug_type": parsed.get("bug_type"),
        "plausibility": parsed.get("plausibility"),
        "rationale": parsed.get("rationale"),
        "raw": raw,
        "parse_error": False,
    }
