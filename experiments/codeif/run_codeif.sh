#!/usr/bin/env bash
# Runs the CodeIF pipeline (Lin et al., arXiv:2502.19166) as shipped.
#
# We do not reimplement any of it: this only supplies the arguments that the
# benchmark's own scripts leave blank (api keys, judge model, model list), and
# calls its three stages in order:
#     src/llm_gen/gen_api_answer.py   generation
#     src/llm_gen/gen_judge.py        LLM-as-judge compliance labelling
#     src/llm_gen/model_result.py     metric aggregation
#
# Usage:
#   ./run_codeif.sh <question_file> <version> <models_csv> [judge_model]
#
# <models_csv> entries are  name[@base_url]  -- omit @base_url to use
# OpenRouter. A local vLLM endpoint is given explicitly, e.g.
#   'Qwen/Qwen2.5-Coder-14B-Instruct@http://127.0.0.1:8000/v1'
#
# Example:
#   ./run_codeif.sh data/question/test_v2_hard_50.jsonl hard50 \
#     'meta-llama/llama-3.1-8b-instruct,openai/gpt-oss-20b'
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${HERE}/CodeIF"
PY=$HOME/miniconda3/envs/mbppeval/bin/python

QFILE="${1:?usage: run_codeif.sh <question_file> <version> <models_csv> [judge_model]}"
VERSION="${2:?}"
MODELS_CSV="${3:?}"
JUDGE_MODEL="${4:-deepseek/deepseek-v4-flash}"   # cheap modern judge; paper used openai/gpt-4o-2024-11-20

OR_BASE="https://openrouter.ai/api/v1"
OR_KEY="$(cat $HOME/.openrouter_key)"
LANG=en
MAXTOK=4096
PARALLEL="${PARALLEL:-16}"

cd "${BENCH}"
export PYTHONPATH="${BENCH}/src"

ANSDIR="output/answer/${VERSION}"
JUDGEDIR="output/judge/${VERSION}"
mkdir -p "${ANSDIR}" "${JUDGEDIR}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
DONE_MODELS=()

for spec in "${MODELS[@]}"; do
  MODEL="${spec%%@*}"
  if [[ "${spec}" == *"@"* ]]; then BASE="${spec#*@}"; KEY="not-needed"
  else BASE="${OR_BASE}"; KEY="${OR_KEY}"; fi

  echo "=== [gen] ${MODEL}  (${BASE})  $(date +%H:%M:%S) ==="
  ${PY} src/llm_gen/gen_api_answer.py \
      --model "${MODEL}" --api_base "${BASE}" --api_key "${KEY}" \
      --questions_file "${QFILE}" --output_dir "${ANSDIR}" \
      --max_tokens ${MAXTOK} --parallel ${PARALLEL} --language_type ${LANG} \
    || { echo "  [gen] FAILED for ${MODEL}, skipping"; continue; }

  echo "=== [judge] ${MODEL} by ${JUDGE_MODEL}  $(date +%H:%M:%S) ==="
  ${PY} src/llm_gen/gen_judge.py \
      --api_base "${OR_BASE}" --api_key "${OR_KEY}" \
      --gen_model "${MODEL}" --judge_model "${JUDGE_MODEL}" \
      --answer_file "${ANSDIR}/${MODEL}.jsonl" --output_dir "${JUDGEDIR}" \
      --max_tokens ${MAXTOK} --parallel ${PARALLEL} --language_type ${LANG} \
    || { echo "  [judge] FAILED for ${MODEL}, skipping"; continue; }

  DONE_MODELS+=("${MODEL}")
done

if [ ${#DONE_MODELS} -eq 0 ]; then echo "no models completed"; exit 1; fi

NAMES=$(IFS=,; echo "${DONE_MODELS[*]}")
echo "=== [aggregate] ${NAMES} ==="
${PY} src/llm_gen/model_result.py \
    --gen_model_list "${NAMES}" --judge_model "${JUDGE_MODEL}" \
    --output_dir "output/model_result/${VERSION}" --version "${VERSION}"

echo "=== done $(date) ==="
find "output/model_result/${VERSION}" -type f 2>/dev/null | sed 's|^|  |'
