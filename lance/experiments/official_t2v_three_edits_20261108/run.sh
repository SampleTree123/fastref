#!/bin/bash
set -euo pipefail
source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate /scratch/prj0000000275/jcy/.conda/envs/xks_lance
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/prj0000000275/jcy/.cache/huggingface
export OMP_NUM_THREADS=8
exec python -u /scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/lance/experiments/official_t2v_three_edits_20261108/run_chain.py "$@"
