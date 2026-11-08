#!/usr/bin/env bash
# GPU evaluation interpreter. Preserve scheduler-assigned CUDA_VISIBLE_DEVICES.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HOME="$ROOT/cache/huggingface" HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/torchinductor"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS=2
exec "$ROOT/envs/qwen/bin/python" "$@"
