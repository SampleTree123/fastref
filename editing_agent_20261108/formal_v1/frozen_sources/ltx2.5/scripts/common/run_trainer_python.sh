#!/usr/bin/env bash
# Isolated native trainer interpreter; never inherit user-site/PYTHONPATH.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../common" && pwd -P)/project_paths.sh"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PATH="$ROOT/runtime/envs/trainer/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/runtime/envs/trainer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/runtime/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/runtime/cache/torch"
export TRITON_CACHE_DIR="$ROOT/runtime/cache/triton"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$ROOT/runtime/envs/trainer/bin/python" "$@"
