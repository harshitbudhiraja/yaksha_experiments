# Proficiency diagnosis (tutor-as-evaluator)

Measures how well a **tutor LLM** can recover a student's latent proficiency
profile from dialogue alone.

A **simulated CS1 student** is conditioned on a ground-truth profile
`P_true: S1..S12 -> {1,2,3,4,5}` over the 12-stage cognitive flow
(`common/stages.py`) and told to act it out without ever naming it. The tutor
sees the problem and the conversation, runs

```
Probe student -> observe response -> update belief
```

for up to `N` turns, and emits a full 12-stage belief vector **every turn**.
The last one is `P_pred`. The tutor never sees `P_true`.

## What's here

| File | Role |
|---|---|
| `../../common/stages.py` | S1–S12 taxonomy, the 1–5 scale, the tutor-facing rubric and the student-facing behavioral anchors |
| `profiles.py` | ground-truth profile generator (7 archetypes, seeded) |
| `student.py` | simulated-student prompt + turn function |
| `tutor.py` | tutor prompt, JSON action schema, tolerant parser |
| `run_rollouts.py` | episode driver → `rollouts.jsonl` |
| `evaluate.py` | metrics → `episode_scores.jsonl` + `summary.json` |

## Run

Serve the tutor and the student (separate models is the normal case; the same
model on both sides is a valid self-play condition, just label it as such):

```bash
vllm serve <tutor-model>   --port 8000
vllm serve <student-model> --port 8001
```

```bash
cd experiments/proficiency_diagnosis

python run_rollouts.py \
  --tutor-model   <tutor-model>   --tutor-base-url   http://localhost:8000/v1 \
  --student-model <student-model> --student-base-url http://localhost:8001/v1 \
  --turns 6 --problems 3 --profiles-per-archetype 2 \
  --conditions interactive one_shot blind \
  --output rollouts.jsonl

python evaluate.py --rollouts rollouts.jsonl \
  --output episode_scores.jsonl --summary summary.json
```

For a fixed-size student population use `--n-profiles` instead, which
apportions across the archetype mix below:

```bash
python run_rollouts.py ... --n-profiles 100 --problems 2 --turns 8 \
  --conditions interactive one_shot blind
```

Episode count is `conditions × profiles × problems`; an interactive episode
costs `2N+1` model calls, `one_shot` costs 2, `blind` costs 1. Start with
`--n-profiles 3 --problems 1 --turns 3` as a smoke test.

`--profile-seed` fixes the student population, so two tutor models compared at
the same seed face exactly the same students.

`--reasoning-effort {low,medium,high}` is passed through to the server for
gpt-oss-style models. Reasoning models need a larger `--max-tokens` (4096+):
chain-of-thought eats the budget, and a truncated turn yields an empty
`content` field.

### On a PBS cluster

`run_gptoss_selfplay.pbs` is a worked example — serves gpt-oss-20b once, points
both roles at it, smoke-tests, then runs the full 100-profile grid:

```bash
qsub run_gptoss_selfplay.pbs   # writes results/gptoss_selfplay/
```

It aborts before the full run if the tutor fails to produce parseable JSON on
more than half of smoke turns, so a format mismatch costs minutes rather than
the whole budget.

**Offline-node prerequisites.** Compute nodes here have no network, so anything
fetched lazily at runtime has to be pre-populated from a login node first:

- **Dataset** — `load_dataset("newfacade/LeetCodeDataset")` once, so it lands in
  the HF cache (the job sets `HF_HUB_OFFLINE=1`).
- **Harmony vocab (gpt-oss only)** — gpt-oss renders prompts through the
  harmony encoding, and `openai_harmony` downloads its vocab on first use.
  Without it the model loads, `/v1/models` lists it, and *every* completion
  returns 500 `HarmonyError: failed to download or load vocab file`. Populate
  the cache once:

  ```bash
  export TIKTOKEN_RS_CACHE_DIR=$HOME/harmony_cache
  python -c "from openai_harmony import load_harmony_encoding, HarmonyEncodingName
  load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)"
  ```

  then export the same `TIKTOKEN_RS_CACHE_DIR` in the job.

**A health check must generate a token.** Polling `/v1/models` until the model
appears is not a readiness check — it passes while generation is completely
broken (exactly what the harmony failure looks like). The job script sends one
real chat completion and aborts if it fails. Keep that probe if you adapt this
script; it is the difference between failing in 4 minutes and failing in 21.

## Ground-truth profiles

Uniform-random profiles are a weak test set — they average out, and "3
everywhere" scores well. `profiles.py` therefore samples seven archetypes,
each targeting a different failure mode of the tutor:

| Archetype | n @ 100 | Shape | What it catches |
|---|---|---|---|
| `uniform_random` | 30 | unstructured | broad coverage of the S1–S12 space |
| `spiky_deficit` | 24 | one stage at 1, rest ~4 | can the tutor find a *single* hole? the sharpest localization test. The deficit is **cycled** across stages (2 per stage at n=100), so per-stage coverage is even rather than luck-of-the-draw |
| `band_split_plan_weak` | 12 | S1–S6 low, S7–S12 high | does it distinguish "can't plan" from "can't code"? |
| `band_split_code_weak` | 12 | S1–S6 high, S7–S12 low | same, reversed |
| `novice` | 10 | ~2 everywhere | range compression at the low end |
| `strong` | 8 | ~4 everywhere | range compression at the high end |
| `flat_mid` | 4 | exactly 3 everywhere | trap for score-compressed tutors (they get this one free). These 4 are **deliberately identical** — repeated draws of the same student give a free test-retest reliability read |

Levels are jittered ±1 (except `flat_mid`) so archetypes aren't constants a
tutor could luck into. At n=100 the level histogram is roughly
1:155, 2:249, 3:247, 4:376, 5:173 — all five levels well represented, which
matters because a tutor that never says 1 or 5 should be penalized.

## Conditions

Run all three; the deltas between them are more informative than any single
number.

- **`interactive`** — the real experiment: full probe/observe/update loop.
- **`one_shot`** — tutor sees only the student's first attempt, then must
  score. `one_shot_mae − interactive_mae` = **the value of probing**.
- **`blind`** — tutor never sees the student, scores from the problem alone.
  This is the prior-only floor. **If `interactive` does not beat `blind`, the
  experiment failed, not the tutor** — it means either the student isn't
  expressing its latent vector or the tutor isn't using the dialogue. Check
  this before reading anything else.

## Metrics

**Localization accuracy** — `localization_mae`, the mean absolute distance
between `P_pred` and `P_true` over the 12 stages.

Reported alongside three references computed on the same profile population,
because raw MAE is not interpretable on its own:

- `best_constant_mae` — the best possible single-number-everywhere answer,
  chosen with hindsight. A hard floor to beat.
- `random_uniform_mae` — random guessing.
- **`skill_over_best_constant`** = `best_constant_mae − localization_mae`.
  **This is the headline number.** `≤ 0` means the tutor extracted no usable
  information, however good its MAE looked.

**Convergence cost** — three definitions, reported separately because they
disagree in useful ways:

- `declared_stop_turn` — when the tutor said it was done. Compare against
  accuracy: stopping at turn 2 with a bad profile is overconfidence, and
  `--ignore-stop` re-runs to the full budget to price what early stopping cost.
- `stabilization_turn` — when the belief vector stopped moving (mean change
  ≤ 0.25/stage for 2 consecutive turns), whether or not the tutor noticed.
- `turns_to_best` — when the error curve first reached its final value.
- `auc_mae` — mean MAE across all turns. The single best summary of
  "accurate *and* fast", since a tutor that is right at turn 6 and a tutor
  that is right at turn 2 share a `localization_mae`.

**Supporting metrics** (these explain *why* a tutor scores as it does):

| Metric | Reads as |
|---|---|
| `spearman_rho` | does it get the profile *shape* right even if the scale is off? high rho + bad MAE = calibration problem, not diagnosis problem |
| `signed_bias` | systematic generosity/harshness (positive = over-rating) |
| `within_one_rate`, `exact_match_rate` | tolerance-band accuracy |
| `weak_recall_at3` | of the student's 3 genuinely weakest stages, how many made the tutor's bottom 3 — the metric that matches what a tutor would actually remediate next |
| `pred_level_spread` | range compression: a tutor only ever saying 3 and 4 has a spread of 1 |
| `per_stage_mae` | which stages are hard to diagnose (expect S4, S5, S12 to be worst — they're the least visible in a transcript) |
| `probe_targeting_counts`, `never_probed_stages` | coverage: stages the tutor scored but never actually probed are guesses |
| `parse_fail_rate` | format compliance (one retry is allowed before a turn counts as failed) |
| `student_leak_rate` | **validity check** — student utterances that name stages, rubrics or self-ratings. Non-zero means the tutor was handed the answer instead of diagnosing it; those episodes are not measuring what you think |

`by_archetype` breaks `interactive` down per profile shape — a tutor with good
overall MAE but `spiky_deficit` MAE near the constant baseline is smoothing,
not diagnosing.

## Known limits

- **The student is the instrument.** Every number here is conditional on the
  student model faithfully expressing `P_true`. `student_leak_rate` and the
  `blind` gap are the built-in validity checks; neither proves fidelity, it
  only detects gross failure. A stronger check (not implemented) is a paired
  run over profiles differing in exactly one stage, confirming the tutor's
  belief moves on that stage and not others.
- **Self-play is confounded.** Running the same model as tutor and student
  (as `run_gptoss_selfplay.pbs` does) shares idiosyncrasies between the two
  roles: the tutor is reading behavior generated by its own distribution, which
  inflates the score by an unknown amount. It is a useful control and a cheap
  first run, but a self-play number is not comparable to a cross-model one.
  Report it as such, and get a cross-model run before drawing conclusions.
- A strong student model may be unable to convincingly play level-1 stages —
  competence leaks through. Expect compressed error at the low end, and
  compare student models before attributing that to the tutor.
- `problems` are drawn from LeetCodeDataset filtered to `Easy`, which is only
  an approximation of CS1 work; swap the loader if you have a real CS1 set.
- Stages are scored independently, but they aren't independent in reality
  (bad S3 forces bad S7). MAE ignores that structure.
