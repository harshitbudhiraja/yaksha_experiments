"""Runs a candidate solution against a LeetCodeDataset row's own tests in an
isolated, resource-limited subprocess (common/sandbox_runner.py).

Extracted out of the original evaluate.py so every experiment (positive-code
eval, bug injection, self-consistency) can get a pass/fail + pass-rate
verdict on arbitrary code without re-implementing the subprocess plumbing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

SANDBOX_RUNNER = Path(__file__).parent / "sandbox_runner.py"

DEFAULT_MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB per problem


def run_in_sandbox(
    row: Dict[str, Any],
    code: str,
    overall_timeout: int = 10,
    per_case_timeout: int = 3,
    wall_timeout: int = 30,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
) -> Dict[str, Any]:
    """Execute `code` against `row["test"]` / `row["input_output"]`.

    Returns a dict with (at least): check_pass, cases_total, cases_passed,
    timed_out, define_error, check_error, case_errors.
    """
    payload = {
        "task_id": row.get("task_id"),
        "prompt": row["prompt"],
        "generated_code": code,
        "test": row["test"],
        "entry_point": row["entry_point"],
        "input_output": row.get("input_output") or [],
        "overall_timeout": overall_timeout,
        "per_case_timeout": per_case_timeout,
        "max_memory_bytes": max_memory_bytes,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        data_path = f.name

    result = {
        "task_id": row.get("task_id"),
        "check_pass": False,
        "cases_total": len(row.get("input_output") or []),
        "cases_passed": 0,
        "timed_out": False,
        "define_error": None,
        "check_error": None,
        "case_errors": [],
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(SANDBOX_RUNNER), data_path],
            capture_output=True,
            text=True,
            timeout=wall_timeout,
        )
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["check_error"] = f"wall-clock timeout after {wall_timeout}s"
        return result
    finally:
        Path(data_path).unlink(missing_ok=True)

    stdout_lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not stdout_lines:
        result["define_error"] = (
            f"sandbox produced no output (exit={proc.returncode}); stderr: {proc.stderr[-2000:]}"
        )
        return result

    try:
        parsed = json.loads(stdout_lines[-1])
    except json.JSONDecodeError:
        result["define_error"] = f"could not parse sandbox output: {stdout_lines[-1][:2000]}"
        return result

    result.update(parsed)
    return result


def compiles(code: str) -> bool:
    """Cheap syntax-only check, no execution. Used to distinguish
    syntax-breaking bugs from logic-only bugs without spinning up a
    subprocess."""
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
