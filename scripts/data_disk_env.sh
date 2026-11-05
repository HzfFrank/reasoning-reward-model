#!/usr/bin/env bash
# Optional storage configuration; any writable disk is supported.
export V7_DATA_ROOT="${V7_DATA_ROOT:-$HOME/.local/share/v7-reward}"
export V7_RUNS_ROOT="${V7_RUNS_ROOT:-$V7_DATA_ROOT/runs}"
[[ "$V7_DATA_ROOT" = /* && "$V7_RUNS_ROOT" = /* ]] || { echo 'V7_DATA_ROOT and V7_RUNS_ROOT must be absolute' >&2; return 2; }
export HF_HOME="${HF_HOME:-$V7_DATA_ROOT/cache/hf}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$V7_DATA_ROOT/cache}"
export TMPDIR="${TMPDIR:-$V7_DATA_ROOT/tmp}" PIP_CACHE_DIR="${PIP_CACHE_DIR:-$V7_DATA_ROOT/cache/pip}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$V7_DATA_ROOT/cache/uv}" TORCH_HOME="${TORCH_HOME:-$V7_DATA_ROOT/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$V7_DATA_ROOT/cache/triton}" CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$V7_DATA_ROOT/cache/cuda}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$V7_DATA_ROOT/cache/extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$V7_DATA_ROOT/cache/inductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$V7_DATA_ROOT/cache/mpl}" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$V7_RUNS_ROOT"
