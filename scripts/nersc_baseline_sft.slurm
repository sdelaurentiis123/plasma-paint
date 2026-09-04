#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --job-name=plasma-paint-sft
#SBATCH --output=artifacts/plasma_painter/job-%j.log
set -euo pipefail
module load pytorch/2.8.0
source .venv/bin/activate
export PLASMA_PAINTER_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1701
git rev-parse HEAD
nvidia-smi
srun python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version baseline_v1
srun python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version optimized_v1
# The SFT entry point refuses updates unless >=20 actual model candidates exist
# and >=80% pass validation. A failed gate ends this allocation without training.
srun python -m plasma_painter.training.sft --config configs/plasma_painter/pilot.yaml --execute
