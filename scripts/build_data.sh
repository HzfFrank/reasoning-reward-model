#!/usr/bin/env bash
# Run on a working copy of raw trajectories: progress and collage files are created there.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${1:?Usage: build_data.sh WORKING_TRAJECTORY_ROOT OUTPUT_DIR}"
: "${2:?Output directory is required}"
: "${V3_BASE_MODEL:?Set V3_BASE_MODEL to local Qwen3-VL-8B-Instruct}"
PYTHON=${PYTHON:-python}
"$PYTHON" - "$1" <<'PY_CHECK'
from pathlib import Path
import sys
from toolkits.libero_sole.common import find_episode_dirs, success_from_episode
items=find_episode_dirs(sys.argv[1])
if not items: raise SystemExit('No episodes found')
bad=[str(p) for p in items if success_from_episode(p) is not True]
if bad: raise SystemExit(f'v3 construction requires confirmed successful episodes: {bad[:5]}')
PY_CHECK
mkdir -p "$2"
"$PYTHON" -m toolkits.libero_sole.compute_contact_progress_v3 "$1" --beta 0.75
"$PYTHON" -m toolkits.libero_sole.generate_cot_dataset_v3 "$1" \
 --sampling-strategy hybrid --num-samples 25 --max-event-samples 5 \
 --from-zero-ratio 0.3 --dropout-seed 20260813 --paraphraser qwen \
 --model-path "$V3_BASE_MODEL" --batch-size 1 --torch-dtype bfloat16 --max-new-tokens 220 \
 --output "$2/sole_cot_dataset.jsonl" --resume \
 --hf-dataset-dir "$2/sole_cot_dataset_hf"
