# Instruction following (bug-injection meta-instructions)

Design doc coverage: **Section 4**, scoped to the meta-instructions that
matter specifically for a bug-injecting tutor -- general-purpose
instruction-following (arbitrary format/style requests) is well covered by
existing IF benchmarks (IFEval etc.) and isn't reproduced here.

Six instruction variants are sent per problem:

| Variant | Compliance check |
|---|---|
| `single_bug` | ≤2 changed lines vs the reference (`common/code_diff.changed_line_numbers`) -- a "single bug" that rewrites half the function doesn't count |
| `syntax_error_requested` | `compiles() == False` |
| `logic_error_requested` | `compiles() == True` |
| `target_line` | the changed line (vs reference) is within ±1 of the requested line number |
| `no_leak_fix` | response has ≤1 code block and no leak phrase ("the fix is", "the bug is", ...) in the prose outside the code block |
| `format_single_block` | exactly one fenced code block and (almost) no prose outside it |

## Run

```bash
python generate.py \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --split test --limit 30 --output if_generations.jsonl

python evaluate.py \
  --generations if_generations.jsonl --split test \
  --output if_results.jsonl --summary if_summary.json
```

`if_summary.json` reports overall + per-variant compliance rate, and
`has_code_rate` (did the model even produce parseable code for that
variant, separate from whether it followed the specific instruction).

## Caveats

- `no_leak_fix`'s phrase list is a fixed set of English hedges
  (`LEAK_PHRASES` in `evaluate.py`); a model that leaks the fix in a way
  that doesn't match any of those phrases will be scored as compliant. This
  is a precision-over-recall choice -- extend the list if you see false
  negatives in `if_results.jsonl`.
- `single_bug`'s ≤2-line threshold is a heuristic proxy for "minimal edit,"
  shared with `bug_injection`'s subtlety metrics -- it doesn't distinguish
  a 2-line bug from two 1-line bugs. `common/code_diff.token_edit_distance`
  gives a finer-grained number if you need it.
- This experiment does not test **robustness to adversarial/pressuring
  prompts** trying to extract the answer (design doc Section 4's third
  bullet) -- that needs multi-turn prompts simulating a student pushing
  back, which isn't implemented here (see the root README's "out of
  scope" section).
