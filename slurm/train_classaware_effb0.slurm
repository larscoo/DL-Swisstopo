#!/bin/bash
#SBATCH --job-name=swisstopo-classaware
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

cd "${HOME}/DL-Swisstopo"
mkdir -p logs

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate deep-learning

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Requires the newer train.py with:
# --augmentation-mode
# --imbalance-strategy
# --threshold-min-recall
python3 train.py \
  --device cuda \
  --model efficientnet_b0 \
  --pretrained on \
  --epochs 12 \
  --batch-size 256 \
  --num-workers 12 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --augmentation-mode class_aware \
  --imbalance-strategy both \
  --threshold-min-recall 0.80 \
  --output-dir artifacts/a100_classaware_both

echo "Job finished at $(date)"
