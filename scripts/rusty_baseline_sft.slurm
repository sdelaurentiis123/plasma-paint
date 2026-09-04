#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=cca
#SBATCH --qos=gen
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=plasma-paint-sft
#SBATCH --output=artifacts/plasma_painter/job-%j.log
set -euo pipefail
module load python/3.11.11
source .venv/bin/activate
export PLASMA_PAINTER_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1701
git rev-parse HEAD
nvidia-smi
srun python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version baseline_v1
srun python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version optimized_v1
srun python -m plasma_painter.training.sft --config configs/plasma_painter/pilot.yaml --execute
