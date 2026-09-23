# Self-consistency & ground-truth reliability

Design doc coverage: **Section 5**. Feeds off `experiments/bug_injection/`
output: takes candidate code the model previously generated (buggy or,
occasionally, accidentally-still-correct) and asks the model to *review it
blind* -- as if auditing a student's submission, not recalling its own
generation. Ground truth for scoring comes from actually running the code
(`common/execution.run_in_sandbox`), not from what the model claimed when
it generated the bug in the first place.

This is the check for whether Yaksha's self-reported claims about its own
bugs (which would otherwise feed downstream to a proctor/grading system)
match reality.

## Run

Requires a `bug_generations.jsonl` from `experiments/bug_injection/generate.py` first:

```bash
cd ../bug_injection
python generate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --split test --limit 30 --output bug_generations.jsonl
cd ../self_consistency

python generate.py \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --bug-generations ../bug_injection/bug_generations.jsonl \
  --split test --output sc_generations.jsonl

python evaluate.py \
  --generations sc_generations.jsonl --split test \
  --output sc_results.jsonl --summary sc_summary.json
```

## Metrics

- **is_buggy accuracy / precision / recall / F1** -- does the model's
  yes/no bugginess claim match `check_pass` from sandbox execution?
- **Line-localization accuracy** -- does the claimed bug line fall within
  ±1 of a line that actually differs from the reference solution?
- **Bug-type verification accuracy** -- does the claimed taxonomy category
  match `common/bug_taxonomy.classify_bug_heuristic`'s classification of
  the actual diff (same heuristic-classifier caveat as `bug_injection/`
  applies here -- see that experiment's README).

All three are reported overall and stratified `by_difficulty`/`by_topic`.

## Note on what "ground truth" means here

- `actual_buggy` = sandbox execution fails at least one assertion in the
  dataset's own `check()`. A generation that's syntactically buggy but
  happens to still satisfy every test case (rare, but possible for weak
  test suites) would count as "not actually buggy" here even if a human
  would call the code wrong -- this experiment measures self-consistency
  against *executable* ground truth, not against human judgment.
- `actual_bug_type` reuses the same heuristic classifier as
  `bug_injection/`, so verification accuracy inherits its blind spots.
