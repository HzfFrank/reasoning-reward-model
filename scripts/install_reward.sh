#!/usr/bin/env bash
# Use the official SOLE-R1 reward_server environment for both v3 and V7.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /path/to/sole-r1-model [--reuse]" >&2; exit 2
fi
SOLE_REPO=$(cd "$1" && pwd)
SERVER="$SOLE_REPO/reward_server"
[[ -f "$SERVER/pyproject.toml" ]] || { echo "Missing SOLE-R1 reward_server/pyproject.toml" >&2; exit 2; }
[[ "${2:-}" = "" || "${2:-}" = --reuse ]] || { echo "Unknown option: $2" >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 2; }
# Adapt an existing official environment; --reuse remains a compatible alias.
PY="${V7_REWARD_PYTHON:-$SERVER/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "Install the official SOLE-R1 environment first, or set V7_REWARD_PYTHON: $PY" >&2; exit 2; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), "This setup is validated with Python 3.10"'
uv pip install --default-index https://pypi.org/simple --python "$PY" -r "$ROOT/environments/reward-core.txt"
"$PY" -c 'import torch, torchvision, transformers, deepspeed, datasets, pyarrow, openai, jsonschema, tomli, yaml, tensorboard; print("SOLE-R1 environment ready for v3/V7:", __import__("sys").executable)'
echo "export V7_REWARD_PYTHON=$PY"
echo "export PYTHON=$PY"
