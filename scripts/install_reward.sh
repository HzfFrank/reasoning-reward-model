#!/usr/bin/env bash
# NVIDIA CUDA 12.8 reference environment; CUDA driver compatibility is required.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${1:-"$ROOT/.venv-reward"}
if [[ -e "$ENV_DIR" ]]; then
  echo "Environment already exists: $ENV_DIR. Choose a new directory; nothing overwritten." >&2
  exit 2
fi
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 2; }
uv venv --python "${V7_INSTALL_PYTHON:-3.10}" "$ENV_DIR"
PY="$ENV_DIR/bin/python"
uv pip install --python "$PY" torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" -r "$ROOT/environments/reward-core.txt" \
  -c "$ROOT/environments/reward-linux-py310.constraints.txt"
uv pip check --python "$PY"
"$PY" -c 'import torch, torchvision, transformers, accelerate, PIL, jsonschema; print("Reward runtime imports OK; CUDA:", torch.cuda.is_available())'
echo "Reward Python: $PY"
