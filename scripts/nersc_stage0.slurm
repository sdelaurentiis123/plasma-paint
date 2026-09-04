#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --job-name=plasma-painter-stage0
#SBATCH --output=artifacts/plasma_painter/logs/stage0-%j.out

set -euo pipefail

: "${PLASMA_PAINTER_VENV:?Set PLASMA_PAINTER_VENV to a prepared environment}"
: "${PLASMA_PAINTER_MODEL_PATH:?Set PLASMA_PAINTER_MODEL_PATH to staged Qwen weights}"
: "${PLASMA_PAINTER_SOURCE_PATH:?Set PLASMA_PAINTER_SOURCE_PATH to the permitted 85604 NPZ}"
: "${PLASMA_PAINTER_GEOMETRY_PATH:?Set PLASMA_PAINTER_GEOMETRY_PATH to the permitted 85604 geometry NPZ}"

source "${PLASMA_PAINTER_VENV}/bin/activate"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=1701
mkdir -p artifacts/plasma_painter/logs

srun --cpu-bind=cores python -m plasma_painter.data.build_manifest --config configs/plasma_painter/pilot.yaml
srun --cpu-bind=cores python -m plasma_painter.features.build_cache --config configs/plasma_painter/pilot.yaml
srun --cpu-bind=cores python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version baseline_v1
srun --cpu-bind=cores python -m plasma_painter.generation.sample_programs --config configs/plasma_painter/pilot.yaml --backend local --count 12 --prompt-version optimized_v1
