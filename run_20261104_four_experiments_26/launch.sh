#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$HERE/../../../.." && pwd -P)"
JOB_ID="${1:?Usage: bash launch.sh ALLOCATED_JOB_ID [GPU_COUNT]}"
GPU_COUNT="${2:-4}"
exec srun --jobid="$JOB_ID" --overlap --exact --nodes=1 --ntasks=1 \
  --cpus-per-task="$((GPU_COUNT * 8))" --gres="gpu:$GPU_COUNT" --mem="$((GPU_COUNT * 150))G" \
  bash "$ROOT/scripts/common/run_trainer_python.sh" "$HERE/run.py" --gpu-count "$GPU_COUNT"
