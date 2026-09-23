# Bug selection (Yaksha bug-injection scheme, simulated learners)

Evaluates the **selection rule** of the bug-injection scheme (paper §"Bug
Injection Scheme"): given a belief over a learner's 12-stage profile, which
bug class β and difficulty d to show next, scored by

```
Score(β,d) = λ1·I − λ2·|π̂ − π*| + λ3·⟨q_β, w_t⟩ − λ4·ρ_t(β)
```

and how the belief evolves as the learner succeeds or fails.

## Learners

The simulated students from `../proficiency_diagnosis` (cross-model runs, 200
interactive episodes per tutor = 100 profiles × 2 problems):

- `p_true`: the ground-truth profile the student model acted out, used to
  sample responses;
- `p_pred` + confidence: the Socratic tutor's estimate after the dialogue,
  used as the **initial belief**.

Initial beliefs (`--inits`):

| init | S1–S6 | S7–S12 |
|---|---|---|
| `socratic_pop` (paper) | tutor's Socratic estimate | population average (cold start) |
| `pop_only` | population average | population average |
| `tutor_all12` | tutor estimate | tutor estimate |
| `oracle` | true profile | true profile (upper bound) |

A tutor estimate becomes a discretised Gaussian around `p_pred`, with σ = 0.6 + 1.2·(1 − c). Confidence c is capped at 0.8. The population average is the per-stage level histogram of `p_true` over the same population, add-one smoothed.

## Simulation, per round

1. Score all 16 × 9 (β, d) pairs (`d ∈ {1, 1.5, …, 5}`); take the argmax.
2. Learner succeeds with P = σ(a_β(θ − d)), where θ = min over the true required stages.
3. Factorised belief update: an exact marginal Bayes step for each required stage, checked against brute-force enumeration.
4. Optional learning (η > 0): a success on a stretch bug (d ≥ s_k* − 1.5) raises the weakest required stage by one level with probability η.

The LLM realiser and the debugging dialogue are abstracted away, so each round is a single Bernoulli draw.

## Policies

`yaksha` (all four terms), one-term ablations (`no_info`, `no_success`,
`no_targeting`, `no_variety`), `info_only` (classic CAT), `success_only`
(pure 85% rule), `random`, and `round_robin` (cycle classes at d = 3).

Each term is min-max normalised across candidates every round and λ = 1 for
all of them, so no single term dominates by scale. `--no-normalize` uses the
raw terms instead.

## Design values (not fitted; see `qmatrix.py`)

- Q-matrix: 16 classes, the 12 representative classes plus 4 overlapping ones.
- Discrimination a_β ∈ [1.0, 1.8].
- Prerequisite graph `PREREQS`.
- Readiness threshold τ = 2.5 on mean level (`--tau-lvl`).
- Repetition penalty: γ = 0.7, W = 5.
- π* = 0.85.

## Run

```bash
python simulate.py --rounds 30 --seeds 5 --output results/main   # ~5 min, CPU
python report.py results/main        # report.md + mae_curves.png
```

## Metrics

| metric | meaning |
|---|---|
| `MAE@T`, `MAE-AUC` | belief posterior mean vs the true profile, at the last round and averaged over rounds |
| `succ`, `\|succ−π*\|` | realised success rate and its per-round gap to π* |
| `ZPD` / `too hard` | share of shown bugs with a true success probability in [0.7, 0.95] / below 0.5 |
| `H(class)`, `max run` | class entropy normalised by log 16; longest run of the same class |
| `weak-hit` | shown class requires one of the learner's 3 truly weakest stages |
| `prereq viol` | shown class requires a stage whose prerequisite is truly below τ |
| `Σ level gain` | total true-level increase over the episode (learning runs only) |
