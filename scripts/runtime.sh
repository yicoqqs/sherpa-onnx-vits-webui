#!/usr/bin/env bash
# Source this file; it only configures this process and directories below VITS_ROOT.
set -euo pipefail
VITS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export VITS_ROOT
if [[ -f "$VITS_ROOT/deploy.env" ]]; then
  set -a
  source "$VITS_ROOT/deploy.env"
  set +a
fi
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export MAMBA_ROOT_PREFIX="$VITS_ROOT/runtime/mamba"
export CONDA_PKGS_DIRS="$VITS_ROOT/runtime/packages"
export PIP_CACHE_DIR="$VITS_ROOT/cache/pip"
export UV_CACHE_DIR="$VITS_ROOT/cache/uv"
export UV_NO_CONFIG=1 UV_PYTHON_DOWNLOADS=never UV_CONCURRENT_DOWNLOADS=8
export XDG_CACHE_HOME="$VITS_ROOT/cache"
export XDG_CONFIG_HOME="$VITS_ROOT/config"
export XDG_DATA_HOME="$VITS_ROOT/data/runtime-data"
export HF_HOME="$VITS_ROOT/cache/huggingface"
export TORCH_HOME="$VITS_ROOT/cache/torch"
export NUMBA_CACHE_DIR="$VITS_ROOT/cache/numba"
export MPLCONFIGDIR="$VITS_ROOT/cache/matplotlib"
export TRITON_CACHE_DIR="$VITS_ROOT/cache/triton"
export PYTHONPYCACHEPREFIX="$VITS_ROOT/cache/pycache"
export TMPDIR="$VITS_ROOT/data/tmp"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_XET=1
export PATH="$VITS_ROOT/runtime/py312/bin:$VITS_ROOT/runtime/bin:$PATH"
export LD_LIBRARY_PATH="$VITS_ROOT/runtime/py312/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$VITS_ROOT/runtime/bin" "$VITS_ROOT/runtime/run" "$VITS_ROOT/runtime/downloads" \
  "$VITS_ROOT/data/logs" "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
  "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$UV_CACHE_DIR" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
VITS_PYTHON="$VITS_ROOT/runtime/py312/bin/python"
VITS_PID_FILE="$VITS_ROOT/runtime/run/vits.pid"
VITS_HOST="${VITS_HOST:-0.0.0.0}"
VITS_PORT="${VITS_PORT:-7860}"
export VITS_HOST VITS_PORT
vits_is_running() {
  [[ -f "$VITS_PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$VITS_PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' '\n' < "/proc/$pid/cmdline" | grep -Fxq -- "$VITS_ROOT/app.py"
}
