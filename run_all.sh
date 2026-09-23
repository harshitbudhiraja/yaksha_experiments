#!/usr/bin/env bash
# Runs every experiment in this repo against one served model and drops all
# outputs under results/<tag>/<experiment>/.
#
# Prerequisite: the model is already being served on an OpenAI-compatible
# endpoint, e.g.:
#   vllm serve <your-model> --port 8000
#
# Usage:
#   ./run_all.sh <base_url> <model_name> <tag> [limit]
#
# Example:
#   ./run_all.sh http://localhost:8000/v1 Qwen/Qwen2.5-Coder-14B-Instruct tuned 30
#
# `limit` caps how many LeetCodeDataset problems each custom experiment uses
# (omit for the full split -- slow; start with a small limit for a smoke test).
set -euo pipefail

BASE_URL="${1:?usage: run_all.sh <base_url> <model_name> <tag> [limit]}"
MODEL="${2:?usage: run_all.sh <base_url> <model_name> <tag> [limit]}"
TAG="${3:?usage: run_all.sh <base_url> <model_name> <tag> [limit]}"
LIMIT="${4:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/results/${TAG}"
mkdir -p "${OUT}"

echo "=== [1/6] solving_ability (lm-eval-harness: humaneval, mbpp, mmlu) ==="
( cd "${ROOT}/experiments/solving_ability" && \
  ./run_lm_eval.sh "${BASE_URL}" "${MODEL}" "${TAG}" )
cp -r "${ROOT}/experiments/solving_ability/results/${TAG}" "${OUT}/solving_ability"

echo "=== [2/6] leetcode_eval (positive code: pass@1 / test-case pass rate) ==="
mkdir -p "${OUT}/leetcode_eval"
( cd "${ROOT}/experiments/leetcode_eval" && \
  python generate_vllm.py --base-url "${BASE_URL}" --model "${MODEL}" \
    --split test --limit "${LIMIT}" --output "${OUT}/leetcode_eval/generations.jsonl" && \
  python evaluate.py --generations "${OUT}/leetcode_eval/generations.jsonl" --split test \
    --output "${OUT}/leetcode_eval/results.jsonl" --summary "${OUT}/leetcode_eval/summary.json" )

echo "=== [3/6] bug_injection (taxonomy hit-rate, closeness, fixability, diversity) ==="
mkdir -p "${OUT}/bug_injection"
( cd "${ROOT}/experiments/bug_injection" && \
  python generate.py --base-url "${BASE_URL}" --model "${MODEL}" \
    --split test --limit "${LIMIT}" --bug-types all --subtlety medium --repeats 3 \
    --output "${OUT}/bug_injection/bug_generations.jsonl" && \
  python evaluate.py --generations "${OUT}/bug_injection/bug_generations.jsonl" --split test \
    --output "${OUT}/bug_injection/bug_results.jsonl" --summary "${OUT}/bug_injection/bug_summary.json" )

echo "=== [4/6] instruction_following (meta-instruction compliance) ==="
mkdir -p "${OUT}/instruction_following"
( cd "${ROOT}/experiments/instruction_following" && \
  python generate.py --base-url "${BASE_URL}" --model "${MODEL}" \
    --split test --limit "${LIMIT}" --output "${OUT}/instruction_following/if_generations.jsonl" && \
  python evaluate.py --generations "${OUT}/instruction_following/if_generations.jsonl" --split test \
    --output "${OUT}/instruction_following/if_results.jsonl" --summary "${OUT}/instruction_following/if_summary.json" )

echo "=== [5/6] multiturn_if (CodeIF-Bench: IA/CA/IFR/CIF across a conversation) ==="
mkdir -p "${OUT}/multiturn_if"
( cd "${ROOT}/experiments/multiturn_if" && \
  [ -d CodeIF-Bench/data ] || ./fetch_data.sh && \
  for MODE in dynamic static; do \
    python run_dialogue.py --base-url "${BASE_URL}" --model "${MODEL}" \
      --mode "${MODE}" --limit "${LIMIT}" \
      --output "${OUT}/multiturn_if/dialogues_${MODE}.jsonl" && \
    python evaluate.py --dialogues "${OUT}/multiturn_if/dialogues_${MODE}.jsonl" \
      --output "${OUT}/multiturn_if/results_${MODE}.jsonl" \
      --summary "${OUT}/multiturn_if/summary_${MODE}.json"; \
  done )

echo "=== [6/6] self_consistency (self-report vs execution ground truth) ==="
mkdir -p "${OUT}/self_consistency"
( cd "${ROOT}/experiments/self_consistency" && \
  python generate.py --base-url "${BASE_URL}" --model "${MODEL}" \
    --bug-generations "${OUT}/bug_injection/bug_generations.jsonl" --split test \
    --output "${OUT}/self_consistency/sc_generations.jsonl" && \
  python evaluate.py --generations "${OUT}/self_consistency/sc_generations.jsonl" --split test \
    --output "${OUT}/self_consistency/sc_results.jsonl" --summary "${OUT}/self_consistency/sc_summary.json" )

echo
echo "All done. Summaries:"
find "${OUT}" -name '*summary*.json' -o -name 'results_*.json' | sort
