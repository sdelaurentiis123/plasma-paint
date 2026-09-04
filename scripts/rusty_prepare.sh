#!/bin/bash
set -euo pipefail
module load python/3.11.11
python -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR="$PWD/.pip-cache" HF_HOME="$PWD/.hf-cache"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.56.2 peft==0.17.1 accelerate==1.10.1 numpy==2.1.2 scipy==1.16.2 scikit-image==0.25.2 Pillow==11.3.0 imageio==2.37.0 matplotlib==3.10.6 PyYAML==6.0.3 huggingface-hub==0.35.3 nodejs-wheel==24.19.0
python scripts/rebase_staged_features.py
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", local_dir="models/Qwen2.5-Coder-7B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"])'
python -m pip freeze > artifacts/plasma_painter/dependencies-rusty.txt
