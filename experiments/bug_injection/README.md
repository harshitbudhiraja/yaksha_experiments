# Bug-injection controllability

Design doc coverage: **Section 2** in full -- the core Yaksha-specific axis
("given a correct solution, inject *this* kind of bug, *this* subtly").
No standard coding benchmark measures this, so unlike `solving_ability/`
this is fully custom.

Uses `common/bug_taxonomy.py`'s 8-category taxonomy (syntax, off-by-one,
logic, semantic/algorithmic, data-structure misuse, edge-case omission,
type error, student misconception) as the fixed vocabulary that makes
"requested vs delivered" measurable at all.

## Run

```bash
python generate.py \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --split test --limit 30 \
  --bug-types all --subtlety medium --repeats 3 \
  --output bug_generations.jsonl

python evaluate.py \
  --generations bug_generations.jsonl --split test \
  --output bug_results.jsonl --summary bug_summary.json
```

Run `generate.py` three times (once per `--subtlety` value) against the
same `--limit`/problems to populate the subtlety-calibration metric
(Section 2.6) with more than one point.

## What's measured, and how

| Metric | Section | How |
|---|---|---|
| Taxonomy hit-rate | 2.1 | `common/bug_taxonomy.classify_bug_heuristic` (diff-pattern heuristic) vs `bug_type_requested`. **Heuristic, not ground truth** -- see caveat below. |
| Edit distance / AST-diff / line similarity | 2.2 | `common/code_diff.py`, comparing `buggy_code` to the dataset's `completion` (reference solution) |
| Pass-rate distribution | 2.3 | Runs `buggy_code` through `common/execution.run_in_sandbox`; buckets into 0% / (0,25%) / [25,75%) / [75,100%) / 100% |
| Fixability | 2.4 | Runs the model's own `fixed_code` (requested alongside the bug) through the sandbox; `fixability_success` = does it now pass everything |
| Diversity/entropy | 2.5 | Shannon entropy of delivered bug type across `--repeats` for the same `task_id` |
| Subtlety calibration | 2.6 | Pearson correlation between requested subtlety (easy/medium/hard, ordinal) and actual pass_rate across generations |
| Safety flag | 8 | Regex-style check for dangerous constructs (`os.system`, `subprocess`, `eval(`, ...) present in `buggy_code` but absent from the reference |

All of the above are also broken out `by_difficulty` and `by_topic`
(Section 0 stratification).

## Important caveat: the heuristic classifier

`classify_bug_heuristic` is a cheap, diff-pattern-based guess (comparison
operator changed -> `off_by_one`, `set`/`heap` disappeared ->
`data_structure_misuse`, etc.). It's good enough for a directional,
always-available signal, but categories like `logic` vs
`semantic_algorithmic` are genuinely fuzzy from a diff alone, and it will
misclassify some fraction of real bugs.

For a more trustworthy number, pass `--use-llm-judge` is **not yet wired
into `evaluate.py`** -- `common/judge.py` has a working
`classify_bug_llm()` you can call in a small patch to `evaluate_one()` if
you want an LLM-judge secondary opinion (also gives you the "plausibility"
rating from Section 2.2's "would a human plausibly write this?" question,
which the heuristic can't answer at all). Left as an opt-in because it
roughly doubles the number of model calls per generation.

## Fixability requires the model to cooperate

`fixability_success` is only computed when the model actually returned a
`fixed_code` block (the prompt asks for one). If your model under test
doesn't reliably follow that part of the format, `fixability_rate` will be
computed over a smaller, biased subset -- check `parse_ok_rate` and the
per-generation `bug_results.jsonl` before trusting it.
