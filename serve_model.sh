#!/usr/bin/env bash
# Serves a model as an OpenAI-compatible endpoint via vLLM.
# Run this ON A GPU NODE (e.g. inside your PBS job) -- not on the login node.
# Once it's up, point run_all.sh (or any experiment script) at
# http://localhost:<port>/v1.
#
# Usage:
#   ./serve_model.sh [model] [port] [tensor-parallel-size]
#
# The candidate models for this suite (pass one explicitly -- there is no
# default, so a forgotten argument can't silently serve the wrong model):
#   meta-llama/Llama-3.1-8B-Instruct        1 GPU
#   Qwen/Qwen2.5-Coder-14B-Instruct         1 GPU
#   openai/gpt-oss-20b                      1 GPU
#   google/gemma-4-31b-it                   tensor-parallel 2 on 40GB cards
#   Qwen/Qwen3-Coder-30B-A3B-Instruct       tensor-parallel 2 on 40GB cards
#
# Example:
#   ./serve_model.sh Qwen/Qwen2.5-Coder-14B-Instruct 8000 1
#   ./serve_model.sh google/gemma-4-31b-it 8000 2
set -euo pipefail

MODEL="${1:?no model given -- name one of the candidate models listed above}"
PORT="${2:-8000}"
TP_SIZE="${3:-1}"

# HF weights (~15GB in bf16) land in $HF_HOME/hub -- point this at scratch/
# project space instead of your home quota if home is small. Uncomment and
# adjust if needed:
# export HF_HOME=/scratch/${USER}/hf_cache

if ! python3 -c "import vllm" >/dev/null 2>&1; then
  echo "vllm not found in this env -- installing..."
  pip install vllm
fi

echo "Serving ${MODEL} on port ${PORT} (tensor-parallel-size=${TP_SIZE})"
vllm serve "${MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
