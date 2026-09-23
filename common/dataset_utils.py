"""Helpers for loading and normalizing newfacade/LeetCodeDataset rows.

Shared by every experiment: dataset schema (columns confirmed against the
HF dataset viewer) is:
    task_id, question_id, difficulty, tags, problem_description,
    starter_code, prompt, completion, entry_point, test, input_output,
    query, response
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional

from datasets import load_dataset

DATASET_NAME = "newfacade/LeetCodeDataset"

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def load_split(split: str = "test", limit: Optional[int] = None):
    """Load a split of the dataset, optionally truncated to `limit` rows."""
    ds = load_dataset(DATASET_NAME, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def build_generation_prompt(row: Dict[str, Any]) -> str:
    """Reproduce the instruct-style prompt the dataset authors used (the
    `query` field), falling back to a manual template if it's missing.
    """
    if row.get("query"):
        return row["query"]

    return (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"### Question:\n{row['problem_description']}\n\n"
        "Please complete the following starter code, and enclose your final "
        "code within a ```python code block.\n"
        f"```python\n{row['starter_code']}\n```\n\n"
        "### Answer: (use the provided format with backticks)"
    )


def get_reference_solution(row: Dict[str, Any]) -> str:
    """The dataset's own canonical/editorial solution, used as the known-good
    baseline that bug-injection experiments mutate."""
    return row.get("completion") or ""


def stratum(row: Dict[str, Any]) -> Dict[str, Any]:
    """Stratification keys every experiment should report metrics by
    (Section 0 of the eval design doc)."""
    tags = row.get("tags") or []
    return {
        "difficulty": row.get("difficulty") or "unknown",
        "topic": tags[0] if tags else "unknown",
        "tags": tags,
    }


def extract_code(text: str) -> str:
    """Pull the first fenced python code block out of a model response,
    falling back to the raw text if no fence is present."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_all_code_blocks(text: str) -> List[str]:
    """All fenced code blocks in a response, in order. Used to detect
    solution leakage (e.g. a second block containing a 'corrected' version)."""
    return [m.strip() for m in CODE_BLOCK_RE.findall(text)]


def iter_rows(ds) -> Iterator[Dict[str, Any]]:
    for row in ds:
        yield row
