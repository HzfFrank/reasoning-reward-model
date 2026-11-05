#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/data_disk_env.sh"
: "${V7_REWARD_PYTHON:?}" "${V7_RUNS_ROOT:?}"
if [[ $# -ne 3 ]]; then echo 'usage: train_success.sh TRAIN_JSONL BASE_MODEL OUTPUT' >&2; exit 2; fi
ARGS=(--data "$1" --model-path "$2" --output-dir "$3")
"$V7_REWARD_PYTHON" "$ROOT/stage7_sft/train_distributed.py" "${ARGS[@]}" --preflight-only
exec "$V7_REWARD_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$ROOT/stage7_sft/train_distributed.py" "${ARGS[@]}"
