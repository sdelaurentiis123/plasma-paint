"""Download public weights into this dedicated run; no plasma files are read."""
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

repo='Qwen/Qwen2.5-VL-7B-Instruct'
revision=HfApi(token=False).model_info(repo).sha
print(f'Public model revision: {revision}',flush=True)
path=Path('models/Qwen2.5-VL-7B-Instruct')
snapshot_download(repo,revision=revision,local_dir=path,token=False,max_workers=2,
                  allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
Path('models/vision-model-manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'path':str(path)},indent=2)+'\n')
print('Model download complete',flush=True)
