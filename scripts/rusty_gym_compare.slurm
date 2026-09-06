#!/bin/bash
#SBATCH --partition=gpupreempt
#SBATCH --account=cca
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=plasma-gym-compare
#SBATCH --output=artifacts/plasma_painter/gym-compare-%j.log
set -euo pipefail
module load python/3.11.11
source .venv/bin/activate
export PYTHONPATH="$PWD/vision-deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
git rev-parse HEAD
srun python -m plasma_painter.evaluation.gym_adapter \
 --model models/Qwen2.5-VL-7B-Instruct --adapter trained-adapter \
 --output artifacts/plasma_painter/gym-adapter-comparison-v1
