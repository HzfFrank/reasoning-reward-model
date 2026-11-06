#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# New environment by default. Existing compatible SOLE-R1 environments can be reused directly.
ENV_DIR=${1:-.venv}
if [ -e "$ENV_DIR" ]; then echo "Environment already exists: $ENV_DIR. Use it directly or choose a new directory." >&2; exit 2; fi
uv venv --python 3.10 "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV_DIR/bin/python" setuptools wheel packaging ninja
DS_BUILD_OPS=0 uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation -r requirements.txt -c requirements.lock.txt
