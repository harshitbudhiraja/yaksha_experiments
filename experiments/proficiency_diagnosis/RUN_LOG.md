# Proficiency diagnosis — run log

Four grids, each varying the **tutor** model against a fixed **student**.
`skill` = `best_constant_mae − interactive_mae`; **positive means the tutor
extracted usable information**. `bias` = signed bias (positive = over-rating).
`spread` = `pred_level_spread` (range compression; 0 = one number for all 12 stages).


## `gptoss_selfplay` — 100 profiles, student = gpt-oss-20b (self-play)

| Tutor | Status | Episodes | blind MAE | one_shot MAE | interactive MAE | skill | bias (int.) | spread (int.) |
|---|---|---|---|---|---|---|---|---|
| gptoss_selfplay | complete | 600 | 1.150 | 1.610 | **1.669** | **-0.601** | +1.560 | 0.15 |

## `cross_model` — 100 profiles, student = gpt-oss-20b (local vLLM)

| Tutor | Status | Episodes | blind MAE | one_shot MAE | interactive MAE | skill | bias (int.) | spread (int.) |
|---|---|---|---|---|---|---|---|---|
| gemma4-31b | complete | 600 | 1.114 | 1.713 | **1.820** | **-0.752** | +1.779 | 0.53 |
| gptoss-20b | complete | 600 | 1.177 | 1.603 | **1.613** | **-0.546** | +1.497 | 0.12 |
| llama31-8b | complete | 600 | 1.162 | 1.337 | **1.509** | **-0.441** | +1.287 | 1.02 |
| qwen25-coder-14b | complete | 600 | 1.113 | 1.747 | **1.778** | **-0.711** | +1.692 | 0.14 |
| qwen3-coder-30b | complete | 599 | 1.580 | 1.173 | **1.493** | **-0.426** | +1.238 | 0.66 |

## `scale1k` — 1000 profiles, student = gpt-oss-20b (local vLLM)

| Tutor | Status | Episodes | blind MAE | one_shot MAE | interactive MAE | skill | bias (int.) | spread (int.) |
|---|---|---|---|---|---|---|---|---|
| gptoss-20b | complete | 3000 | 1.120 | 1.614 | **1.701** | **-0.656** | +1.640 | 0.07 |
| llama31-8b | complete | 2998 | 1.144 | 1.368 | **1.493** | **-0.448** | +1.154 | 1.31 |
| qwen25-coder-14b | complete | 3000 | 1.109 | 1.733 | **1.751** | **-0.706** | +1.661 | 0.16 |
| gemma4-31b | **incomplete** | 1531 rollouts, no summary | — | — | — | — | — | — |
| qwen3-coder-30b | **incomplete** | 822 rollouts, no summary | — | — | — | — | — | — |

## `apistudent100` — 100 profiles, student = deepseek-v4-flash (API)

| Tutor | Status | Episodes | blind MAE | one_shot MAE | interactive MAE | skill | bias (int.) | spread (int.) |
|---|---|---|---|---|---|---|---|---|
| Qwen_Qwen2.5-Coder-14B-Instruct | complete | 300 | 1.192 | 1.747 | **1.625** | **-0.558** | +1.448 | 0.29 |
| google_gemma-4-31B-it | complete | 300 | 1.123 | 1.705 | **1.755** | **-0.688** | +1.720 | 0.77 |
| openai_gpt-oss-20b | complete | 300 | 1.073 | 1.462 | **1.672** | **-0.604** | +1.575 | 0.65 |
| Qwen_Qwen3-Coder-30B-A3B-Instruct | **incomplete** | 0 rollouts, no summary | — | — | — | — | — | — |
| meta-llama_Llama-3.1-8B-Instruct | **incomplete** | 0 rollouts, no summary | — | — | — | — | — | — |


## Summary

- **Completed runs: 12**; every one has `skill < 0`.
- Range of `skill`: -0.752 to -0.426.
- Baselines (shared): `best_constant_mae` = 1.045 (answer "3" everywhere), `random_uniform_mae` = 1.523.
- **`interactive` is worse than `blind` in every completed run.** The README's validity rule — "if `interactive` does not beat `blind`, the experiment failed, not the tutor" — is therefore triggered across the board.
- Swapping the student from gpt-oss-20b to an API model (`deepseek-v4-flash`, grid `apistudent100`) does **not** change the pattern, so the failure is not specific to one student model.


## Incomplete runs

| Job | Grid | Tutor | Exit | Walltime | Progress |
|---|---|---|---|---|---|
| 78134 | scale1k | gemma4-31b | -29 (walltime kill) | 03:01:06 / 03:00:00 | 1530/3000 (51%) |
| 78135 | scale1k | qwen3-coder-30b | 143 (SIGTERM) | 01:23:14 / 03:00:00 | 821/3000 (27%) |
| — | apistudent100 | Llama-3.1-8B-Instruct | — | — | no rollouts |
| — | apistudent100 | Qwen3-Coder-30B-A3B-Instruct | — | — | no rollouts |

Both scale1k jobs need `walltime=06:00:00`; they were sized for 3h and the 1000-profile grid takes ~5h30m–5h45m at observed rates.
