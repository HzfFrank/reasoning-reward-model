#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${V3_SFT_MODEL:?Set V3_SFT_MODEL to the local v3 SFT checkpoint}"
: "${V3_DATASET:?Set V3_DATASET to the HF save_to_disk directory}"
PYTHON=${PYTHON:-python}
NAME=${RUN_NAME:-rlvr_$(date +%Y%m%d_%H%M%S)}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-4}" \
 stage3_tools/libero_rlvr_train_grpo_v3.py --mode train --run-name "$NAME" \
 --policy-checkpoint "$V3_SFT_MODEL" --dataset "$V3_DATASET" \
 --steps 500 --group-size 8 --temperature 0.7 --top-p 0.9 --max-new-tokens 192 \
 --clip-epsilon 0.2 --kl-beta 0.01 --tau 5 --lr 5e-7 --loss-type grpo \
 --importance-sampling-level token --scale-rewards group --prompt-pool-size 0 \
 --eval-samples 16 --eval-group-size 4 --eval-every-steps 50 --export-every-steps 100 \
 --export-final --tensorboard "$@"
