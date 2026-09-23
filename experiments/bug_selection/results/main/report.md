# Bug-selection simulation — main

Tutors (learner source): gemma4-31b, gptoss-20b, llama31-8b, qwen25-coder-14b, qwen3-coder-30b; 200 learners/tutor; T=30 rounds; 5 seeds; π*=0.85. Pooled over tutors unless noted.

## 1. Policies (init = socratic_pop, learning η=0.0)

| policy | MAE@T | MAE-AUC | succ | |succ-π*| | ZPD | too hard | H(class) | max run | weak-hit | prereq viol |
|---|---|---|---|---|---|---|---|---|---|---|
| yaksha | 1.08 | 1.13 | 0.71 | 0.15 | 0.41 | 0.06 | 0.71 | 1.0 | 0.50 | 0.45 |
| no_info | 1.09 | 1.15 | 0.81 | 0.11 | 0.40 | 0.02 | 0.76 | 1.0 | 0.45 | 0.45 |
| no_success | 1.06 | 1.12 | 0.53 | 0.32 | 0.19 | 0.26 | 0.70 | 1.0 | 0.52 | 0.44 |
| no_targeting | 1.07 | 1.13 | 0.70 | 0.16 | 0.39 | 0.07 | 0.66 | 1.0 | 0.45 | 0.43 |
| no_variety | 1.13 | 1.16 | 0.68 | 0.19 | 0.39 | 0.05 | 0.31 | 16.5 | 0.60 | 0.47 |
| info_only | 1.18 | 1.19 | 0.50 | 0.35 | 0.12 | 0.13 | 0.00 | 30.0 | 0.46 | 0.36 |
| success_only | 1.06 | 1.15 | 0.75 | 0.13 | 0.48 | 0.10 | 0.71 | 4.9 | 0.34 | 0.45 |
| random | 1.04 | 1.13 | 0.43 | 0.42 | 0.19 | 0.53 | 0.90 | 2.0 | 0.44 | 0.48 |
| round_robin | 0.98 | 1.09 | 0.42 | 0.43 | 0.29 | 0.48 | 0.99 | 1.0 | 0.43 | 0.48 |


## 2. Initial belief (policy = yaksha, η=0.0)

| init | MAE@0 | MAE@T | MAE-AUC | succ |
|---|---|---|---|---|
| socratic_pop | 1.26 | 1.08 | 1.13 | 0.71 |
| pop_only | 1.08 | 0.86 | 0.95 | 0.73 |
| tutor_all12 | 1.44 | 1.21 | 1.29 | 0.53 |
| oracle | 0.14 | 0.21 | 0.19 | 0.71 |


## 3. Per tutor (socratic_pop, yaksha vs random, η=0.0)

| tutor | MAE@0 | MAE@T yaksha | MAE@T random | MAE@0 pop_only | succ yaksha |
|---|---|---|---|---|---|
| gemma4-31b | 1.32 | 1.15 | 1.10 | 1.08 | 0.71 |
| gptoss-20b | 1.25 | 1.07 | 1.03 | 1.08 | 0.71 |
| llama31-8b | 1.21 | 1.02 | 1.00 | 1.08 | 0.72 |
| qwen25-coder-14b | 1.30 | 1.12 | 1.08 | 1.08 | 0.71 |
| qwen3-coder-30b | 1.20 | 1.03 | 0.99 | 1.08 | 0.71 |


## 4. With learning (η=0.3, init = socratic_pop)

| policy | Σ level gain | MAE@T | MAE-AUC | succ | H(class) | weak-hit |
|---|---|---|---|---|---|---|
| yaksha | 4.74 | 1.05 | 1.11 | 0.79 | 0.72 | 0.51 |
| no_info | 3.24 | 0.97 | 1.09 | 0.86 | 0.76 | 0.46 |
| no_success | 4.53 | 1.08 | 1.12 | 0.61 | 0.71 | 0.52 |
| no_targeting | 4.62 | 1.09 | 1.12 | 0.79 | 0.67 | 0.44 |
| no_variety | 4.69 | 1.05 | 1.12 | 0.80 | 0.40 | 0.59 |
| info_only | 3.15 | 1.16 | 1.18 | 0.67 | 0.00 | 0.46 |
| success_only | 3.15 | 0.93 | 1.06 | 0.80 | 0.69 | 0.36 |
| random | 2.61 | 0.99 | 1.10 | 0.45 | 0.89 | 0.43 |
| round_robin | 3.33 | 0.98 | 1.08 | 0.45 | 0.99 | 0.43 |
