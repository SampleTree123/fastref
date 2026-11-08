#!/usr/bin/env bash
#SBATCH --job-name=qwen-ltx-editing-agent
#SBATCH --account=prj0000000275
#SBATCH --partition=testqueue
#SBATCH --qos=highpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=1024G
#SBATCH --time=18:00:00
#SBATCH --chdir=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8
#SBATCH --output=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8/experiments/editing_agent_20261108/slurm_%j.log
#SBATCH --error=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8/experiments/editing_agent_20261108/slurm_%j.log
set -euo pipefail
PROJECT_ROOT=/scratch/prj0000000275/jcy/xks/DataMake/10_research/model_experiments/qwen3.8
EDITING_OUTPUT="${EDITING_OUTPUT:-$PROJECT_ROOT/experiments/editing_agent_20261108/formal_v1}"
test -f "$EDITING_OUTPUT/freeze_manifest.json"
# Runs only after this eight-GPU job is allocated. Signal our old numeric srun
# step if needed, keep its allocation intact, and wait before opening outputs.
bash "$PROJECT_ROOT/scripts/run_gpu_python.sh" "$PROJECT_ROOT/scripts/editing_agent/handoff.py" \
  --output "$EDITING_OUTPUT" --timeout 120
srun --nodes=1 --ntasks=1 --cpus-per-task=112 --gres=gpu:h200:8 --mem=1024G --cpu-bind=none \
  bash "$PROJECT_ROOT/scripts/run_gpu_python.sh" "$PROJECT_ROOT/scripts/editing_agent/run_campaign.py" \
  --output "$EDITING_OUTPUT" --gpus 8 --hours 17
