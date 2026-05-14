#!/bin/bash
#SBATCH --job-name=swisstopo-both24
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=students
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

cd "${HOME}/DL-Swisstopo"
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${HOME}/miniforge3"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${HOME}/miniconda3"
else
  echo "Could not locate conda installation." >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate deep-learning

echo "Job started on $(hostname) at $(date)"
nvidia-smi

python3 train.py \
  --device cuda \
  --model efficientnet_b0 \
  --pretrained on \
  --epochs 24 \
  --batch-size 256 \
  --num-workers 12 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --augmentation-mode standard \
  --imbalance-strategy both \
  --output-dir artifacts/runs/server/completed/both_long_lr1e4_effb0

echo "Job finished at $(date)"
