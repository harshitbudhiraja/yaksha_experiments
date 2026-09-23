"""Runs a candidate function against free-standing assert snippets.

`common/execution.py` is specific to LeetCodeDataset rows (a `check(candidate)`
harness plus input/output pairs). CodeIF-Bench instead ships each instruction
with a list of self-contained snippets -- `assert f(x) == y`, a `try/except
ValueError` block, an `inspect.getsource` + `mccabe` complexity check, a
`pycodestyle` style check -- so it needs a different runner.

Each snippet is appended to the candidate code in its own file and run as its
own subprocess: snippet-level granularity (upstream concatenates every snippet
of a requirement into one script and blames all of them when the script exits
non-zero), and one snippet crashing the interpreter can't take the others down.

Note this deliberately does NOT apply `sandbox_runner.reliability_guard`: the
benchmark's own checks need `inspect.getsource` (so the code must live in a
real file, not an `exec`'d string) and need `open`/`os.remove` (pycodestyle
writes a scratch file). Isolation here is a throwaway cwd, a memory cap and a
wall-clock timeout -- weaker than `common/execution.py`. Run untrusted model
output through it only where you'd be willing to run the model's code at all.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

DEFAULT_MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB per snippet


def _limit_memory(max_memory_bytes: int):
    def _apply() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        except Exception:
            pass

    return _apply


def run_snippet(
    code: str,
    snippet: str,
    timeout: int = 15,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
) -> Dict[str, Any]:
    """Run `code` + `snippet` as one script. Returns {pass, error}."""
    if not code.strip():
        return {"pass": False, "error": "no candidate code"}

    with tempfile.TemporaryDirectory() as workdir:
        script = os.path.join(workdir, "candidate_case.py")
        with open(script, "w") as f:
            f.write(code.rstrip() + "\n\n" + snippet.rstrip() + "\n")
        try:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=workdir,  # pycodestyle-style checks write scratch files here
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_memory(max_memory_bytes),
            )
        except subprocess.TimeoutExpired:
            return {"pass": False, "error": f"timeout after {timeout}s"}
        except Exception as exc:  # spawn failure -- report, don't abort the run
            return {"pass": False, "error": f"{type(exc).__name__}: {exc}"}

    if proc.returncode == 0:
        return {"pass": True, "error": None}
    return {"pass": False, "error": (proc.stderr or proc.stdout or "").strip()[-2000:]}


def run_snippets(
    code: str,
    snippets: List[str],
    timeout: int = 15,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Run every snippet; a requirement counts as satisfied only if all of its
    snippets pass. Returns {pass, n_passed, n_total, failed, first_error}."""
    if not snippets:
        return {"pass": False, "n_passed": 0, "n_total": 0, "failed": [], "first_error": "no unit tests"}

    def _one(s: str) -> Dict[str, Any]:
        return run_snippet(code, s, timeout=timeout, max_memory_bytes=max_memory_bytes)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(_one, snippets))
    else:
        results = [_one(s) for s in snippets]

    failed = [s for s, r in zip(snippets, results) if not r["pass"]]
    first_error: Optional[str] = next((r["error"] for r in results if not r["pass"]), None)
    return {
        "pass": not failed,
        "n_passed": len(snippets) - len(failed),
        "n_total": len(snippets),
        "failed": failed,
        "first_error": first_error,
    }
