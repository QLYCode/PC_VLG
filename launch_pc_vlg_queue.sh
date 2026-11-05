#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/queue_gpu3
exec "$PY" -u queue_pc_vlg.py --gpu 3 "$@"
