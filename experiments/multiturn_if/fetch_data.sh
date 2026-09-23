#!/usr/bin/env bash
# Fetches the CodeIF-Bench task files (a few hundred KB of JSONL) next to this
# experiment. Only the `data/` directory is used -- the upstream inference and
# metric scripts are reimplemented here against common/model_client.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/CodeIF-Bench"

if [ -d "${DEST}" ]; then
  echo "${DEST} already exists -- pulling."
  git -C "${DEST}" pull --ff-only
else
  git clone --depth 1 https://github.com/zhu-zhu-ding/CodeIF-Bench.git "${DEST}"
fi

echo
echo "Task files:"
ls -la "${DEST}/data"
