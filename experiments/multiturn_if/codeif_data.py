"""Loader for CodeIF-Bench (arXiv:2503.22688) task files.

The benchmark ships three levels under its `data/` directory:

  L_1_part_1.jsonl      function-level (MBPP-derived, "Non-SA"): a standalone
                        prompt + base asserts + 7 verifiable requirements,
                        every unit test self-contained. 50 tasks.
  L_1_part_2.jsonl      MISNAMED: despite the L_1 prefix these 20 rows carry the
                        repository-level schema (namespace/project_path/
                        completion_path, `tests` as pytest node ids). They are
                        rejected by the loader -- see _is_function_level.
  L_2.jsonl / L_3.jsonl repository-level (DevEval-derived, "SA"): 9
                        requirements whose tests are pytest node ids inside a
                        real project checkout.

Only L_1 is loadable here. L_2/L_3 need the original repositories (a Figshare
archive linked from the CodeIF-Bench README) plus a working per-project test
environment, and their requirements are verified by running `pytest <node id>`
against that checkout -- infrastructure this repo doesn't have. `load_tasks`
says so explicitly rather than silently scoring repo-level rows against
snippets that can't import their own project.

The `multi-turn` field gives the order the benchmark intends the requirements
to be introduced in; `Functionality Extension` is excluded by default because
at function level it deliberately changes the signature (e.g. "add an `all=True`
parameter"), which invalidates the base asserts every other metric is measured
against -- upstream's dynamic-conversation script drops it for the same reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Requirement types, in the benchmark's own naming. The first seven appear at
# function level; the last two only at repository level.
FUNCTION_LEVEL_TYPES = [
    "Input-Output Conditions",
    "Exception Handling",
    "Edge Case Handling",
    "Functionality Extension",
    "Annotation Coverage",
    "Code Complexity",
    "Code Standard",
]
REPO_ONLY_TYPES = [
    "Context Usage Verification",
    "Context Usage Correctness Verification",
]

# Signature-changing requirement: satisfying it breaks the base asserts.
EXTENSION_TYPE = "Functionality Extension"

BASE_INSTRUCTION = "Base"


@dataclass
class Requirement:
    name: str
    instruction: str
    unit_tests: List[str]


@dataclass
class CodeIFTask:
    task_id: str
    level: str
    prompt: str
    base_tests: List[str]
    requirements: Dict[str, Requirement]
    turn_order: List[str]
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def issue_sequence(self) -> List[str]:
        """Base turn first, then the benchmark's intended instruction order."""
        return [BASE_INSTRUCTION] + list(self.turn_order)

    def tests_for(self, name: str) -> List[str]:
        if name == BASE_INSTRUCTION:
            return self.base_tests
        return self.requirements[name].unit_tests

    def instruction_for(self, name: str) -> str:
        if name == BASE_INSTRUCTION:
            return self.prompt
        return self.requirements[name].instruction


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_function_level(row: Dict[str, Any]) -> bool:
    """True only for genuinely self-contained function-level rows.

    `L_1_part_2.jsonl` is misnamed: its rows carry the repository-level schema
    and their `test_list` holds pytest NODE IDS, not runnable asserts. Executed
    as Python, a bare node-id string is a valid no-op that exits 0, so every
    such row would score a perfect sweep on every instruction and silently
    inflate every metric. Require the function-level marker (`task_id` plus a
    `test` list) and reject anything carrying repo-level fields.
    """
    if any(k in row for k in ("namespace", "project_path", "completion_path")):
        return False
    return row.get("task_id") is not None and bool(row.get("test"))


def _to_task(row: Dict[str, Any], level: str, include_extension: bool) -> CodeIFTask:
    requirements = {
        name: Requirement(name=name, instruction=spec["requirement"], unit_tests=list(spec.get("unit_test") or []))
        for name, spec in (row.get("requirements") or {}).items()
    }

    turn_order = [k for k in (row.get("multi-turn") or []) if k in requirements]
    # Fall back to the dict order if `multi-turn` is missing or unusable.
    if not turn_order:
        turn_order = list(requirements)
    if not include_extension:
        turn_order = [k for k in turn_order if k != EXTENSION_TYPE]

    return CodeIFTask(
        task_id=str(row.get("task_id")),
        level=level,
        prompt=row["prompt"],
        base_tests=list(row.get("test") or row.get("test_list") or []),
        requirements=requirements,
        turn_order=turn_order,
        raw=row,
    )


def load_tasks(
    data_dir: str,
    level: str = "L1",
    limit: Optional[int] = None,
    include_extension: bool = False,
) -> List[CodeIFTask]:
    """Load CodeIF-Bench tasks from a checkout of the benchmark's `data/` dir."""
    if level != "L1":
        raise NotImplementedError(
            f"level {level!r} is repository-level (DevEval-derived): its unit tests are "
            "pytest node ids that only run inside the original project checkout. Fetch "
            "the repositories from the Figshare link in the CodeIF-Bench README and wire "
            "up a per-project test environment first -- see this experiment's README."
        )

    root = Path(data_dir)
    paths = sorted(root.glob("L_1_part_*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no L_1_part_*.jsonl under {root}. Run ./fetch_data.sh, or point "
            "--data-dir at a CodeIF-Bench checkout's data/ directory."
        )

    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))

    rejected = [r for r in rows if not _is_function_level(r)]
    rows = [r for r in rows if _is_function_level(r)]
    if rejected:
        print(f"note: skipped {len(rejected)} repository-level rows (see "
              f"_is_function_level); {len(rows)} function-level tasks loaded",
              file=__import__("sys").stderr)
    if limit is not None:
        rows = rows[:limit]

    return [_to_task(r, level, include_extension) for r in rows]
