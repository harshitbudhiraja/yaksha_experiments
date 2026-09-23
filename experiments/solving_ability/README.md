# Solving ability (standard benchmarks, via lm-evaluation-harness)

Design doc coverage: **Section 1** (Core Coding Competence) and the OOD
half of **Section 7** (Robustness & Generalization).

Rather than re-implementing HumanEval/MBPP generation+scoring (already
solved, and easy to get subtly wrong -- prompt formatting, pass@k
estimator, sandboxing), this experiment wraps
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
which already ships `humaneval`, `mbpp`, and `mmlu` tasks and talks to any
OpenAI-compatible server (vLLM's `vllm serve`, in our case).

Custom code in this repo (`experiments/leetcode_eval/`, `bug_injection/`,
etc.) is reserved for the axes no standard harness measures.

## Setup

```bash
pip install "lm_eval[api]"
```

## Run

Point at your served model (vLLM) and give the run a tag:

```bash
./run_lm_eval.sh http://localhost:8000/v1 Qwen/Qwen2.5-Coder-14B-Instruct tuned
```

To get the Section-1 "delta vs base" comparison, also serve and run the
untuned base model (e.g. on a different port) and tag it `base`:

```bash
./run_lm_eval.sh http://localhost:8001/v1 Qwen/Qwen2.5-Coder-14B-Instruct base
```

Each run writes `results/<tag>/**/results_*.json` (+ per-sample outputs via
`--log_samples`).

## Compare

```bash
python compare_base_vs_tuned.py --base results/base --tuned results/tuned
```

Prints a per-task, per-metric delta table and writes `base_vs_tuned.json`.
A large negative delta on `humaneval`/`mbpp` is the pedRL-style regression
signal ("did pedagogical RL cost solving ability?"); a large negative delta
on `mmlu` with no code-related justification is a catastrophic-forgetting
signal.

## Notes

- `humaneval`/`mbpp` execute model-generated code as part of scoring --
  that execution happens inside lm-eval-harness's own process, which is why
  `run_lm_eval.sh` passes `--confirm_run_unsafe_code`. This is a different
  (weaker) isolation boundary than `common/sandbox_runner.py` used
  elsewhere in this repo; only run it against a model you trust isn't
  actively adversarial, same caveat as the rest of this repo.
- Swap/add tasks freely -- `humanevalplus` and `mbppplus` (stricter,
  augmented test suites) are drop-in replacements if installed
  (`pip install "lm_eval[api,zeno]"` pulls in the plus-suite dependencies
  in some lm-eval versions; check `lm_eval --tasks list` if a task 404s).
- This experiment does **not** cover LeetCodeDataset (not an lm-eval task);
  that's what `experiments/leetcode_eval/` is for, run separately.
