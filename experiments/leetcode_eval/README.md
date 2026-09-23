# LeetCodeDataset x Qwen2.5-Coder-14B eval pipeline

Generates solutions with Qwen2.5-Coder-14B (served via vLLM) for every problem
in [newfacade/LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset),
then runs each solution against the dataset's own tests in an isolated,
resource-limited subprocess and reports pass@1 and test-case pass rate.

Run all commands below from this directory (`experiments/leetcode_eval/`).

## Setup (on a CUDA machine)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../../requirements.txt
pip install vllm
```

## 1. Start the model server

```bash
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --port 8000
```

## 2. Smoke test (do this first)

Run a handful of problems through both stages before committing to the full
228-problem split — this catches server/connectivity/parsing issues in
seconds instead of after a long full run:

```bash
python generate_vllm.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --split test \
  --limit 5 \
  --output smoke_generations.jsonl

python evaluate.py \
  --generations smoke_generations.jsonl \
  --split test \
  --output smoke_results.jsonl \
  --summary smoke_summary.json
```

Sanity-check `smoke_summary.json`: `total_problems` should be 5, and you
should not see `define_error`/`check_error` messages that indicate a broken
pipeline (e.g. connection errors, empty `generated_code`, import failures)
rather than the model simply getting a problem wrong. Once that looks right,
move on to the full run below.

## 3. Generate solutions (full run)

```bash
python generate_vllm.py \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --split test \
  --concurrency 8 \
  --output generations.jsonl
```

- Add `--resume` to continue an interrupted run (skips task_ids already in
  `generations.jsonl`).

## 4. Evaluate

```bash
python evaluate.py \
  --generations generations.jsonl \
  --split test \
  --workers 8 \
  --output results.jsonl \
  --summary summary.json
```

This prints (and writes to `summary.json`):

- `pass@1_rate` — fraction of problems where the model's solution satisfies
  every assertion in the dataset's `check()` function (the standard
  code-eval metric).
- `test_case_pass_rate` — fraction of individual `input_output` test cases
  passed across the whole dataset (finer-grained than pass@1).
- A breakdown by `difficulty` (Easy/Medium/Hard).

`results.jsonl` has one line per problem with `check_pass`, `cases_total`,
`cases_passed`, and any error messages, for drilling into specific failures.

## Sandboxing notes

Each problem's generated code runs in its own subprocess
(`sandbox_runner.py`), with:
- A memory cap (`RLIMIT_AS`, default 1 GiB) and a hard wall-clock kill
  timeout (`--wall-timeout`, default 30s) plus a separate SIGALRM-based
  timeout for the overall `check()` call and for each individual test case.
- Destructive builtins (`os.remove`, `shutil.rmtree`, `subprocess.Popen`,
  process signals, etc.) disabled before the generated code executes.

This is process-level isolation suitable for local experimentation with a
model you trust not to be actively adversarial — it is not a hardened
security sandbox. For untrusted/adversarial code, run this inside a
container or VM instead.

## Files

- `generate_vllm.py` — calls the vLLM OpenAI-compatible server to generate
  one solution per problem, extracts the code block, writes `generations.jsonl`.
- `evaluate.py` — orchestrates sandboxed runs across all generations,
  aggregates pass@1 and test-case pass rate, writes `results.jsonl` +
  `summary.json`.

Dataset loading (`common/dataset_utils.py`), the sandbox subprocess
(`common/sandbox_runner.py` + `common/execution.py`), and the model client
(`common/model_client.py`) live in the repo-root `common/` package, shared
with the other experiments under `experiments/`.

This experiment is the "positive code" / core-solving-ability slice of the
full eval suite — see the [repo-root README](../../README.md) for how it
maps to the other experiments (bug injection, instruction following,
self-consistency, and the standard-benchmark `solving_ability` suite).
