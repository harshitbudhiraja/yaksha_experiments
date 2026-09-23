#!/usr/bin/env bash
# Vendors the CodeIF benchmark (Lin et al., arXiv:2502.19166) as-is.
# Only its data/ and src/ are used; we do not modify the benchmark code, so
# the numbers this experiment produces are the benchmark's own metrics.
#
# NOTE: distinct from experiments/multiturn_if/, which vendors CodeIF-*Bench*
# (Ding et al., arXiv:2503.22688) -- confusingly similar name, different paper.
set -euo pipefail
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/CodeIF"
[ -d "${DEST}" ] && { echo "already present: ${DEST}"; exit 0; }
git clone --depth 1 https://github.com/lin-rany/codeIF.git "${DEST}"
rm -rf "${DEST}/.git"
echo "fetched CodeIF -> ${DEST}"
