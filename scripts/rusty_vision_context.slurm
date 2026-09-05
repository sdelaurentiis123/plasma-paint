#!/bin/bash
#SBATCH --partition=gpupreempt
#SBATCH --account=cca
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=plasma-paint-vision
#SBATCH --output=artifacts/plasma_painter/job-%j.log
set -euo pipefail
module load python/3.11.11
source .venv/bin/activate
export PYTHONPATH="$PWD/vision-deps"
export PLASMA_PAINTER_MODEL_PATH="$PWD/models/Qwen2.5-VL-7B-Instruct"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1701
git rev-parse HEAD
nvidia-smi
python -m pip freeze > artifacts/plasma_painter/dependencies-vision.txt
python -m pip freeze --path vision-deps >> artifacts/plasma_painter/dependencies-vision.txt
srun python -m plasma_painter.generation.vision_context
