#!/usr/bin/env bash
# Standard-benchmark solving-ability run via lm-evaluation-harness, pointed at
# any OpenAI-compatible chat endpoint (vLLM's `vllm serve ...`).
#
# Covers design doc:
#   Section 1 (HumanEval/MBPP pass@1/pass@k) and
#   Section 7 (OOD retention via MMLU, to catch catastrophic forgetting
#              from pedagogical RL fine-tuning).
#
# Requires: pip install "lm_eval[api]"
#
# Usage:
#   ./run_lm_eval.sh <base_url> <model_name> <tag> [extra lm_eval args...]
#
# Example (tuned model):
#   ./run_lm_eval.sh http://localhost:8000/v1 Qwen/Qwen2.5-Coder-14B-Instruct tuned
# Example (base model, for the Section-1 delta comparison):
#   ./run_lm_eval.sh http://localhost:8001/v1 Qwen/Qwen2.5-Coder-14B-Instruct base
set -euo pipefail

BASE_URL="${1:?usage: run_lm_eval.sh <base_url> <model_name> <tag>}"
MODEL_NAME="${2:?usage: run_lm_eval.sh <base_url> <model_name> <tag>}"
TAG="${3:?usage: run_lm_eval.sh <base_url> <model_name> <tag>}"
shift 3 || true

OUT_DIR="results/${TAG}"
mkdir -p "${OUT_DIR}"

# humaneval/mbpp execute model-generated code to score it -- that's lm-eval's
# job here, not common/sandbox_runner.py's; --confirm_run_unsafe_code
# acknowledges that. mmlu is a held-out non-code benchmark used purely as an
# OOD-retention / catastrophic-forgetting check (Section 7).
lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/chat/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
  --tasks humaneval,mbpp,mmlu \
  --gen_kwargs "temperature=0.2,max_gen_toks=1536" \
  --confirm_run_unsafe_code \
  --output_path "${OUT_DIR}" \
  --log_samples \
  "$@"

echo "Results written to ${OUT_DIR}"
