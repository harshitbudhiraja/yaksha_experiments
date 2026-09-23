"""Explicit bug taxonomy for the bug-injection controllability axis
(design doc Section 2.1). Defining this up front is what makes
"requested-bug-type vs delivered-bug-type accuracy" measurable at all --
without a fixed vocabulary there's nothing to score hit-rate against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BugType:
    id: str
    label: str
    description: str
    request_instruction: str  # phrasing used when *asking* the model for this bug
    # substrings/heuristics used by the heuristic classifier (best-effort only,
    # see classify_bug_heuristic below)


BUG_TYPES: List[BugType] = [
    BugType(
        id="syntax",
        label="Syntax error",
        description="Missing colon/bracket, bad indentation -- code fails to parse.",
        request_instruction="Introduce a syntax error (e.g. a missing colon, mismatched "
        "bracket, or broken indentation) so the code fails to parse/compile.",
    ),
    BugType(
        id="off_by_one",
        label="Off-by-one / boundary error",
        description="Wrong comparison operator or index bound (< vs <=, start/end offset).",
        request_instruction="Introduce an off-by-one / boundary error, e.g. change a `<` to "
        "`<=`, or shift a loop/index bound by one.",
    ),
    BugType(
        id="logic",
        label="Logic error",
        description="Wrong operator, swapped condition, or wrong variable used.",
        request_instruction="Introduce a logic error: swap a condition, use the wrong "
        "variable, or use the wrong operator, while keeping the code syntactically valid.",
    ),
    BugType(
        id="semantic_algorithmic",
        label="Semantic/algorithmic error",
        description="A subtly wrong variant of the intended algorithm (not a full rewrite).",
        request_instruction="Introduce a subtle algorithmic error -- implement a nearly-"
        "correct but flawed variant of the intended algorithm (e.g. wrong traversal order, "
        "wrong recurrence), while preserving the overall structure.",
    ),
    BugType(
        id="data_structure_misuse",
        label="Data-structure misuse",
        description="Uses the wrong data structure for the job (e.g. list instead of set/heap).",
        request_instruction="Introduce a data-structure misuse bug: replace the correct data "
        "structure (e.g. a set or heap) with an inappropriate one (e.g. a list) that changes "
        "behavior or complexity.",
    ),
    BugType(
        id="edge_case_omission",
        label="Edge-case omission",
        description="Doesn't handle empty input, None, or negative numbers.",
        request_instruction="Introduce an edge-case omission bug: remove or break the "
        "handling of an edge case such as empty input, None, or negative numbers.",
    ),
    BugType(
        id="type_error",
        label="Type error / implicit conversion bug",
        description="Wrong type assumption or implicit conversion (e.g. str vs int).",
        request_instruction="Introduce a type error: cause an incorrect implicit type "
        "conversion or a wrong type assumption (e.g. string vs int, float division vs int).",
    ),
    BugType(
        id="student_misconception",
        label="Common student misconception",
        description="Mutable default argument, reference-vs-value copy, integer division, etc.",
        request_instruction="Introduce a bug that mirrors a common student misconception, "
        "such as a mutable default argument, confusing reference copy with value copy, or "
        "using integer division where float division was intended.",
    ),
]

BUG_TYPES_BY_ID: Dict[str, BugType] = {b.id: b for b in BUG_TYPES}


def get(bug_id: str) -> Optional[BugType]:
    return BUG_TYPES_BY_ID.get(bug_id)


# --- heuristic classifier -------------------------------------------------
# Best-effort, diff-pattern-based classification of *what kind* of bug was
# actually introduced, used to score taxonomy hit-rate automatically. This is
# a heuristic, not a ground truth oracle -- categories like
# "semantic_algorithmic" vs "logic" are inherently fuzzy from a diff alone.
# For a rigorous number, run the same generations through common/judge.py
# (LLM-judge) and treat this as a cheap, always-available secondary signal.

COMPARISON_OPS = ["<=", ">=", "<", ">", "=="]


def classify_bug_heuristic(reference: str, buggy: str, compiles_ok: bool) -> str:
    if not compiles_ok:
        return "syntax"

    import difflib

    ref_lines = reference.splitlines()
    buggy_lines = buggy.splitlines()
    sm = difflib.SequenceMatcher(a=ref_lines, b=buggy_lines)
    changed_ref, changed_new = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed_ref.extend(ref_lines[i1:i2])
            changed_new.extend(buggy_lines[j1:j2])

    ref_blob = "\n".join(changed_ref)
    new_blob = "\n".join(changed_new)

    if any(op in ref_blob for op in COMPARISON_OPS) and any(op in new_blob for op in COMPARISON_OPS):
        # a comparison operator changed -> likely off-by-one
        if ref_blob.replace("<=", "<").replace(">=", ">") != new_blob.replace("<=", "<").replace(">=", ">") \
                or ref_blob != new_blob:
            for a, b in (("<=", "<"), (">=", ">"), ("<", "<="), (">", ">=")):
                if a in ref_blob and b in new_blob:
                    return "off_by_one"

    if "range(" in ref_blob or "range(" in new_blob:
        return "off_by_one"

    if any(kw in new_blob for kw in ("set(", "{}", "heapq", "deque")) != \
            any(kw in ref_blob for kw in ("set(", "{}", "heapq", "deque")):
        return "data_structure_misuse"

    if ("None" in ref_blob or "not " in ref_blob or "len(" in ref_blob) and \
            ("None" not in new_blob and "not " not in new_blob):
        return "edge_case_omission"

    if "//" in new_blob and "//" not in ref_blob:
        return "student_misconception"
    if "int(" in new_blob and "int(" not in ref_blob:
        return "type_error"

    if not changed_ref and not changed_new:
        return "logic"

    # default bucket for changes that don't match a sharper pattern
    return "logic"
