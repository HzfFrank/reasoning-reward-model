#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/data_disk_env.sh"
: "${V7_REWARD_PYTHON:?}" "${V7_RUNS_ROOT:?}"
if [[ $# -ne 1 ]]; then echo 'usage: train_mixed.sh PREPARED_RUN_DIR' >&2; exit 2; fi
"$V7_REWARD_PYTHON" "$ROOT/stage7_sft/train_distributed_success_failure.py" --experiment-dir "$1" --preflight-only
exec "$V7_REWARD_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$ROOT/stage7_sft/train_distributed_success_failure.py" --experiment-dir "$1"
