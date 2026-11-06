#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${V3_BASE_MODEL:?Set V3_BASE_MODEL to the local Qwen3-VL-8B-Instruct directory}"
: "${V3_DATASET:?Set V3_DATASET to the HF save_to_disk directory}"
PYTHON=${PYTHON:-python}
RUNS=${V3_RUNS_ROOT:-$PWD/runs}
NAME=${RUN_NAME:-sft_$(date +%Y%m%d_%H%M%S)}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-4}" \
 stage2a_tools/libero_sft_e2e.py train-export --dataset "$V3_DATASET" --run-name "$NAME" \
 --train-samples 0 --val-samples 0 --max-optimizer-steps 0 --micro-batch-size 1 \
 --gradient-accumulation-steps 16 --lr 1e-5 \
 --checkpoint-dir "$RUNS/sft/checkpoints/$NAME" --export-dir "$RUNS/sft/models/$NAME" "$@"
