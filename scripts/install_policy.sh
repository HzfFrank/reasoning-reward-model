#!/usr/bin/env bash
# Delegate to the checked-out RLinf installer, then reconcile its package stages.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -lt 1 ]]; then echo 'usage: install_policy.sh RLINF_REPO [ENV_DIR] [official installer flags...]' >&2; exit 2; fi
RLINF=$(cd "$1" && pwd)
shift
ENV_DIR=${1:-"$RLINF/.venv-policy"}
if [[ $# -gt 0 ]]; then shift; fi
if [[ -e "$ENV_DIR" ]]; then echo "Environment already exists: $ENV_DIR; choose a new directory." >&2; exit 2; fi
mkdir -p "$(dirname "$ENV_DIR")"
ENV_DIR=$(cd "$(dirname "$ENV_DIR")" && pwd)/$(basename "$ENV_DIR")
export UV_DEFAULT_INDEX=https://pypi.org/simple
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ENV_DIR/libero_config}"
# Do not create ENV_DIR before RLinf's uv venv step.
(
  cd "$RLINF"
  bash requirements/install.sh embodied --model openpi --env libero \
    --venv "$ENV_DIR" --python 3.11.14 --torch 2.7.1 --no-flash-attn "$@"
)
PY="$ENV_DIR/bin/python"
uv pip install --python "$PY" -r "$ROOT/environments/policy-compat.txt"
# This upstream distribution intentionally supplies patched transformers files.
# Install it last, after the matching generic distribution metadata.
uv pip install --python "$PY" --reinstall --no-deps rlinf-transformer-openpi==4.53.2
uv pip check --python "$PY"
echo "Policy Python: $PY"
echo "Use LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH when running this environment."
