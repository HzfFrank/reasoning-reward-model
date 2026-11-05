#!/usr/bin/env bash
: "${V7_DATA_ROOT:?set V7_DATA_ROOT to your absolute data disk}"
[[ "$V7_DATA_ROOT" = /* ]] || { echo 'V7_DATA_ROOT must be absolute' >&2; return 2; }
export HF_HOME="$V7_DATA_ROOT/cache/hf" XDG_CACHE_HOME="$V7_DATA_ROOT/cache"
export TMPDIR="$V7_DATA_ROOT/tmp" PIP_CACHE_DIR="$V7_DATA_ROOT/cache/pip"
export UV_CACHE_DIR="$V7_DATA_ROOT/cache/uv" TORCH_HOME="$V7_DATA_ROOT/cache/torch"
export TRITON_CACHE_DIR="$V7_DATA_ROOT/cache/triton" CUDA_CACHE_PATH="$V7_DATA_ROOT/cache/cuda"
export TORCH_EXTENSIONS_DIR="$V7_DATA_ROOT/cache/extensions"
export TORCHINDUCTOR_CACHE_DIR="$V7_DATA_ROOT/cache/inductor"
export MPLCONFIGDIR="$V7_DATA_ROOT/cache/mpl" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
