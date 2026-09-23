"""Closeness metrics between a reference (correct) solution and a mutated
(buggy) one -- design doc Section 2.2. A good injected bug is a 1-3 token
change that preserves the algorithm's shape, not a rewrite.
"""
from __future__ import annotations

import ast
import difflib
from typing import List, Optional, Tuple


def token_edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over whitespace-ish tokens (cheap stand-in for a
    real tokenizer -- good enough to detect "rewrite vs minimal edit")."""
    a_tok, b_tok = a.split(), b.split()
    n, m = len(a_tok), len(b_tok)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a_tok[i - 1] == b_tok[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def changed_line_numbers(a: str, b: str) -> List[int]:
    """1-indexed line numbers in `b` that differ from `a` (insertions/
    replacements). Used both as a "how many lines changed" signal and to
    check whether a requested target line was actually the one touched."""
    sm = difflib.SequenceMatcher(a=a.splitlines(), b=b.splitlines())
    changed = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changed.extend(range(j1 + 1, j2 + 1))
    return changed


def line_similarity_ratio(a: str, b: str) -> float:
    """difflib ratio in [0, 1]; 1.0 == identical. A minimal single-bug edit
    on a nontrivial solution should score close to 1.0."""
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def ast_diff_ratio(a: str, b: str) -> Optional[float]:
    """Structural similarity via AST dump comparison. Returns None if either
    snippet fails to parse (e.g. a syntax-error bug -- there's no AST to
    compare in that case; use compiles() + this together)."""
    try:
        dump_a = ast.dump(ast.parse(a))
        dump_b = ast.dump(ast.parse(b))
    except SyntaxError:
        return None
    return difflib.SequenceMatcher(a=dump_a, b=dump_b).ratio()


def diff_summary(a: str, b: str) -> Tuple[int, int, Optional[float], float]:
    """(token_edit_distance, num_changed_lines, ast_diff_ratio, line_ratio)."""
    return (
        token_edit_distance(a, b),
        len(changed_line_numbers(a, b)),
        ast_diff_ratio(a, b),
        line_similarity_ratio(a, b),
    )
