# Multi-turn code instruction following (CodeIF-Bench)

Design doc coverage: **Section 4**, the multi-turn half that
[`experiments/instruction_following/`](../instruction_following/README.md)
explicitly doesn't cover ("robustness across a conversation ... isn't
implemented here"). That experiment scores one-shot compliance with
bug-injection meta-instructions; this one scores whether a model can keep
satisfying instructions as a conversation piles them up.

Built on **CodeIF-Bench** ([arXiv:2503.22688](https://arxiv.org/abs/2503.22688),
[repo](https://github.com/zhu-zhu-ding/CodeIF-Bench)): every instruction ships
with executable unit tests, so instruction following is measured by running
code, not by asking a judge model. The upstream inference/metric scripts are
reimplemented here (they hardcode hosted-API keys and a fixed model name);
only the benchmark's task files are used as-is.

## Why it's in this suite

A bug-injecting tutor is a multi-turn agent: "inject one bug" → "no, make it
subtler" → "now target line 7" → "don't reveal the fix". The failure mode that
matters isn't getting the current turn wrong, it's satisfying turn *n* by
quietly undoing turn *n-2*. CodeIF-Bench's requirement types (I/O contract,
exception behaviour, edge cases, annotations, complexity bound, PEP 8) are a
clean, verifiable proxy for that, and the `forgetting_by_age` curve this
experiment reports says whether the loss is about distance in context or about
instruction type.

## Setup

```bash
./fetch_data.sh                 # clones the benchmark; only data/ is used
pip install mccabe pycodestyle  # the benchmark's own complexity/style checks import these
```

## Run

```bash
# 1. dialogues (model calls + per-turn execution, interleaved)
python run_dialogue.py \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --mode dynamic --limit 20 --output mtif_dynamic.jsonl

# 2. metrics (pure post-processing of the log -- re-scoring is free)
python evaluate.py --dialogues mtif_dynamic.jsonl \
  --output mtif_results.jsonl --summary mtif_summary.json
```

`--mode` picks the interaction protocol; run all three against the same model
to separate capability from context management:

| Mode | Protocol | What it isolates |
|---|---|---|
| `dynamic` | Adaptive, as upstream: each round sends the first still-unsatisfied instruction alone; on failure the model gets one feedback turn quoting the failing check, then that instruction is dropped | Instruction following *with* error-driven repair -- closest to a real session |
| `static` | The benchmark's fixed `multi-turn` order, one instruction per turn, no feedback, no retries | Forgetting on its own, with repair removed |
| `single_turn` | All instructions stated up front in one prompt | The no-conversation control. **Not part of the default sweep** -- `run_all.sh` runs `dynamic` and `static` only, since single-shot instruction following is covered by [`experiments/instruction_following/`](../instruction_following/README.md). Still available via `--mode single_turn` if you want the context-management baseline on these specific instructions |

After every assistant turn, *all* of the task's instructions are re-executed
against the current code -- including ones not yet issued -- so `evaluate.py`
derives every metric offline from one log.

## Candidate models

The five models this axis is run against (serve each with `../../serve_model.sh`,
then point `--base-url`/`--model` at it):

| Model | Notes for this experiment |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 1 GPU |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 1 GPU |
| `openai/gpt-oss-20b` | Puts its answer in a reasoning channel and can leave `content` null; `common/model_client.py` already falls back to `reasoning_content`/`reasoning`. Budget ~2x the turns of a same-size dense model |
| `google/gemma-4-31b-it` | ~62 GB in bf16 -- needs `--tensor-parallel-size 2` on 40 GB cards. Gemma chat templates reject a system role; `model_client.chat_messages` folds it into the first user turn (this experiment sends no system prompt, so it shouldn't trigger) |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | MoE, ~3B active -- fast to decode but still ~60 GB of weights to hold |

Run one model at a time unless you have spare GPUs; a full 70-task, 3-mode
sweep is roughly 25-90 min per model (inference-bound, not execution-bound).

## Metrics

With `S_t` = instructions issued up to turn *t* (the base prompt counts as
one), `i_t` = the instruction turn *t* addressed, `P_t` = instructions issued
earlier that the model had already satisfied:

| Metric | Definition | Reads as |
|---|---|---|
| `IA` | 1 if all unit tests of `i_t` pass at *t* | did it do what was just asked |
| `CA` | `|{n ∈ S_t : n passes at t}| / |S_t|` | how much of the conversation still holds |
| `IFR` | `|{n ∈ P_t : n fails at t}| / |P_t|` | of what it had right, how much it broke |
| `CIF` | instructions still passing on the final code, over *all* of the task's instructions (per dialogue) | how many survived to the end |

**Compare `IA` only within a mode.** `dynamic` selects, by construction, the
instructions the current code *fails*, so its `IA` is conditioned on a harder,
self-selected subset than `static`, which asks about everything in order. The
cross-mode comparison to make is `CIF` / `final_CA` / `IFR`. `CIF` counts every
instruction the task defines for the same reason -- an issued-only count
(kept as `CIF_issued`) would penalise `dynamic` for never asking about the
instructions it already satisfied.

Plus two things upstream doesn't report:

- **`forgetting_by_age`** -- `IFR` split by how many turns ago the forgotten
  instruction was issued. The context-management curve. A rising curve is
  distance-driven forgetting; a flat one means the loss is about which
  instruction, not how long ago.
- **`unprompted_pass_rate`** -- the share of instructions that already pass on
  turn 1, before being asked for. Read `CA` net of this: several requirement
  types are satisfied by any reasonable solution (see caveats), and they
  inflate `CA` without the model following anything.

`by_turn[*].upstream_ife` reproduces the reference implementation's
`IFE = CA_t / t` so numbers stay comparable with the paper's repo. It decays
mechanically with turn index (a ratio divided by a turn count), so prefer
`CIF` and `final_CA`.

Also reported: `IA_first_attempt` vs `IA_after_feedback` per instruction type
(how much a pasted error message is worth), `forget_rate_once_satisfied` per
type, and `base_retained_rate` (does the function still pass its original
MBPP asserts at the end -- the plain regression signal).

## Deviations from the upstream implementation

Called out because they change the numbers:

1. **Snippet-level attribution.** Upstream concatenates a requirement's unit
   tests into one script and, on a non-zero exit, marks all of them failed.
   Here each snippet runs as its own subprocess, so `n_tests_passed` is real
   and one crash can't condemn the rest.
2. **A turn with no runnable code keeps the previous code.** Upstream's
   extractor falls back to treating raw prose as the candidate, which scores a
   chatty non-answer as forgetting every instruction at once. See
   `_extract_candidate` in `run_dialogue.py`; bare text is accepted only if it
   parses.
3. **Blank lines are preserved.** Upstream strips them before testing, which
   is self-defeating on the PEP 8 requirement (`E302` wants blank lines
   between definitions).
4. **`CIF` is computed as its README defines it** (instructions still followed
   at the end), not as `CA_t / t`; the latter is kept alongside as
   `upstream_ife`.
5. **`Functionality Extension` is excluded by default** (`--include-extension`
   to keep it). At function level it deliberately changes the signature, e.g.
   "add an `all=True` parameter", which invalidates the base asserts every
   other metric is measured against. Upstream's dynamic script drops it for
   the same reason, but its static script doesn't.

## Caveats

- **Some benchmark unit tests pass vacuously.** Several `Exception Handling`
  snippets are a bare `try: f(...) \n except ValueError as e: assert ...` with
  no failure branch, so a function that raises nothing passes. Likewise
  `Code Complexity` asserts `get_code_complexity(src) <= 3`, and mccabe's
  helper returns a *violation count*, not a complexity score, so it's almost
  always 0. These are upstream's tests, left unmodified for comparability --
  `unprompted_pass_rate` and `by_instruction_type` exist so you can see which
  types are free points before quoting a `CA`.
- **Weaker sandboxing than the rest of this repo.** `common/assert_runner.py`
  runs candidate code from a real file in a throwaway cwd with a memory cap
  and a timeout, but without `sandbox_runner.reliability_guard`: the
  benchmark's checks need `inspect.getsource` (so the code must be a file, not
  an `exec`'d string) and pycodestyle writes a scratch file. Don't point it at
  output you wouldn't be willing to execute.
- **Only the function level (`L_1`) runs.** `L_2`/`L_3` are the repository-level
  (DevEval-derived) splits, with two extra requirement types
  (`Context Usage Verification`, `Context Usage Correctness Verification`) whose
  tests are pytest node ids inside real project checkouts. Running them needs
  the repository archive (Figshare link in the upstream README) plus a working
  per-project test environment; `--level L2` raises a clear error rather than
  scoring snippets that can't import their own project. Wiring it up means
  replacing `run_snippets` with a `pytest <node id>` call in the checkout.
- **Cost.** Each turn re-executes every instruction's tests, so a dialogue is
  ~`turns × instructions` subprocesses. That's what makes `evaluate.py` free to
  re-run; tune `--test-concurrency`/`--concurrency`, and start with a small
  `--limit`.
