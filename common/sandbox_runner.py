#!/usr/bin/env python3
"""Runs INSIDE an isolated subprocess (spawned by evaluate.py). It is given
a path to a JSON file describing one problem + one model-generated solution,
executes the solution against the dataset's own tests, and prints a single
JSON line with the results.

Never invoke this directly on untrusted code outside of a throwaway process
you're fine with resource-limiting; it does not sandbox filesystem access
beyond disabling the obviously destructive calls.
"""
from __future__ import annotations

import builtins
import json
import signal
import sys
from typing import Optional


class TimeoutException(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutException("timed out")


def reliability_guard(max_memory_bytes: Optional[int] = None) -> None:
    """Best-effort hardening of the process before running arbitrary
    model-generated code: disable process/filesystem-destroying calls and
    cap memory. Mirrors the guard used by common code-eval harnesses
    (e.g. OpenAI HumanEval's execution.py).
    """
    import os
    import shutil
    import subprocess as _subprocess

    if max_memory_bytes is not None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (max_memory_bytes, max_memory_bytes))
        except Exception:
            pass

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    for name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir",
                 "setuid", "fork", "forkpty", "killpg", "rename", "renames", "truncate",
                 "replace", "unlink", "fchmod", "fchown", "chmod", "chown", "chroot",
                 "lchflags", "lchmod", "lchown", "getcwd", "chdir"):
        if hasattr(os, name):
            setattr(os, name, None)

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    _subprocess.Popen = None  # type: ignore[assignment]

    builtins.help = None

    import sys as _sys

    _sys.modules["ipdb"] = None
    _sys.modules["joblib"] = None


def run(data: dict) -> dict:
    result = {
        "task_id": data.get("task_id"),
        "define_error": None,
        "check_pass": False,
        "check_error": None,
        "cases_total": 0,
        "cases_passed": 0,
        "case_errors": [],
    }

    namespace: dict = {}
    try:
        exec(data["prompt"] + "\n" + data["generated_code"], namespace)
    except BaseException as e:
        result["define_error"] = f"{type(e).__name__}: {e}"
        return result

    try:
        candidate = eval(data["entry_point"], namespace)
    except BaseException as e:
        result["define_error"] = f"entry_point lookup failed: {type(e).__name__}: {e}"
        return result

    signal.signal(signal.SIGALRM, _alarm_handler)

    # 1. Whole-problem pass@1 signal via the dataset's own check() harness.
    try:
        exec(data["test"], namespace)
        check_fn = namespace["check"]
        signal.alarm(max(1, int(data.get("overall_timeout", 10))))
        try:
            check_fn(candidate)
            result["check_pass"] = True
        finally:
            signal.alarm(0)
    except BaseException as e:
        result["check_error"] = f"{type(e).__name__}: {e}"

    # 2. Granular per-test-case pass count via input_output pairs.
    io_cases = data.get("input_output") or []
    per_case_timeout = max(1, int(data.get("per_case_timeout", 3)))
    result["cases_total"] = len(io_cases)

    for case in io_cases:
        try:
            case_ns = dict(namespace)
            exec(f"__args = dict({case['input']})", case_ns)
            expected = eval(case["output"], case_ns)

            signal.alarm(per_case_timeout)
            try:
                actual = candidate(**case_ns["__args"])
            finally:
                signal.alarm(0)

            if actual == expected:
                result["cases_passed"] += 1
            else:
                result["case_errors"].append(f"mismatch: got {actual!r} expected {expected!r}")
        except BaseException as e:
            result["case_errors"].append(f"{type(e).__name__}: {e}")

    return result


def main() -> None:
    data_path = sys.argv[1]
    with open(data_path) as f:
        data = json.load(f)

    reliability_guard(max_memory_bytes=data.get("max_memory_bytes"))
    result = run(data)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
