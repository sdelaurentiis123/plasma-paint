#!/bin/bash
# Run only inside a new, dedicated plasma-paint execution directory.
set -euo pipefail
module load pytorch/2.8.0
python -m venv --system-site-packages .venv
source .venv/bin/activate
export PIP_CACHE_DIR="$PWD/.pip-cache"
export HF_HOME="$PWD/.hf-cache"
python -m pip install peft==0.17.1 nodejs-wheel==24.19.0
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", local_dir="models/Qwen2.5-Coder-7B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"])'
python -m pip freeze > artifacts/plasma_painter/dependencies-nersc.txt
