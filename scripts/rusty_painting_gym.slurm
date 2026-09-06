#!/bin/bash
#SBATCH --partition=gpupreempt
#SBATCH --account=cca
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=plasma-paint-gym
#SBATCH --output=artifacts/plasma_painter/gym-job-%j.log
set -euo pipefail
module load python/3.11.11
source .venv/bin/activate
export PYTHONPATH="$PWD/vision-deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
git rev-parse HEAD
python -m pip freeze > artifacts/plasma_painter/gym-dependencies.txt
srun python -m plasma_painter.training.gym_rollout --section 18 --frame-offset 0 \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --reference reference-images/28560.jpg --reference reference-images/14586.jpg \
  --output artifacts/plasma_painter/gym-vision-v1
