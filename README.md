# Yaksha experiments

Evaluation code for LLMs used as CS1 coding tutors. Models are served locally
with vLLM and called through an OpenAI-compatible endpoint.

## Folder structure

```
common/                     shared code: model client, sandboxed execution, bug taxonomy, 12-stage definitions, judge
experiments/
  solving_ability/          HumanEval / MBPP / MMLU via lm-evaluation-harness
  leetcode_eval/            pass@1 and test-case pass rate on LeetCodeDataset
  bug_injection/            can the model inject a requested bug type into correct code
  bug_selection/            which bug to show next, simulated with synthetic learners (no LLM)
  instruction_following/    single-turn tutor constraints (one bug, target line, no fix leakage)
  multiturn_if/             instruction retention across turns (CodeIF-Bench)
  codeif/                   single-turn code instruction following (CodeIF)
  self_consistency/         does the model's own bug report match execution ground truth
  proficiency_diagnosis/    can a tutor recover a simulated student's 12-stage proficiency
run_all.sh                  runs 6 experiments against one served model (not bug_selection, codeif, proficiency_diagnosis)
serve_model.sh              starts a vLLM server
```

Each experiment folder (except `codeif/`) has a README with exact commands, and a `results/`
folder. The `*.pbs` files are PBS job scripts that run one model each.

## Setup

```bash
pip install -r requirements.txt
./serve_model.sh <model> 8000 [tp-size]   # on a GPU node
export no_proxy=localhost,127.0.0.1   # needed behind a proxy
```

`codeif/` and `multiturn_if/` use third-party benchmarks that aren't in this
repo. Run `./fetch_data.sh` in each of those folders once to download them.

## Run

```bash
./run_all.sh http://localhost:8000/v1 <model> <tag> [limit]
```

Output goes to `results/<tag>/`. Start with a small `limit` to smoke-test the setup.
