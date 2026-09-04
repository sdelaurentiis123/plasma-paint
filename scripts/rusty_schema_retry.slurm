#!/bin/bash
#SBATCH --partition=gpupreempt
#SBATCH --account=cca
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=plasma-paint-schema
#SBATCH --output=artifacts/plasma_painter/job-%j.log
set -euo pipefail
module load python/3.11.11
source .venv/bin/activate
export PLASMA_PAINTER_MODEL_PATH="$PWD/models/Qwen2.5-Coder-7B-Instruct"
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=1701
git rev-parse HEAD
nvidia-smi
srun python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 4 --prompt-version schema_v2
srun python -m plasma_painter.renderer.render_program --config configs/plasma_painter/pilot.yaml --limit 4 --clip-count 1
# Diagnostic retry only. No SFT/DPO/GRPO update or automatic larger run.
