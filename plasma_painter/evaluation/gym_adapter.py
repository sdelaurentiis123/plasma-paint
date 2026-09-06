"""Matched full episodes, adapter disabled/enabled; no weight updates."""
import argparse
import json
import time
from pathlib import Path
from contextlib import nullcontext
from collections import Counter
from plasma_painter.config import sha256_file,git_state,write_json
from plasma_painter.training.painting_gym import PaintingGym
from plasma_painter.training.gym_rollout import vision_rollout


def load_frames(cache):
    root=Path(cache);index=json.loads((root/'index.json').read_text());frames=[]
    for y in (0,18,31):
        record=next(r for r in index['records'] if r['y']==y)
        path=root/f'section-y{y}.generated.json'
        if sha256_file(path)!=record['sha256']:raise ValueError('Cache hash mismatch')
        clip=json.loads(path.read_text())
        if clip['split']!='art_train':raise ValueError('Only art_train')
        frame=next(f for f in clip['frames'] if f['source']['frame_index']==7)
        PaintingGym(frame);frames.append((y,frame))
    return frames,index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Refuse to overwrite comparison')
    frames,index=load_frames('artifacts/plasma_painter/free_sections')
    refs=json.loads(Path('reference-images/manifest.json').read_text())['records']
    selected=[r for r in refs if r['artist_title']=='Vincent van Gogh'];assert len(selected)==2
    paths=[]
    for r in selected:
        path=Path('reference-images')/Path(r['file']).name
        assert r['is_public_domain'] and sha256_file(path)==r['sha256'];paths.append(str(path))
    adapter=Path(a.adapter)
    record={'git':git_state(),'source_index':index,'adapter_sha256':sha256_file(adapter/'adapter_model.safetensors'),
        'model':a.model,'seed':1701,'frame':7,'references':selected,'records':[],
        'status':'loading','training_performed':False,'full_fidelity_evaluated':False}
    out.mkdir(parents=True);write_json(out/'manifest.json',record)
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False,
        min_pixels=128*28*28,max_pixels=128*28*28)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False);model.eval()
    record['status']='in_progress';write_json(out/'manifest.json',record)
    for y,frame in frames:
        for policy in ('base','sft'):
            gym=PaintingGym(frame,seed=1701);started=time.perf_counter()
            progress=out/f'{policy}-y{y}-progress.json'
            with model.disable_adapter() if policy=='base' else nullcontext():
                vision_rollout(gym,a.model,paths,loaded=(model,processor),
                    on_step=lambda e:write_json(progress,{'turn':e.turn,'events':e.events}))
            directory=gym.save(out/f'{policy}-y{y}')
            obs=gym.observe()
            item={'policy':policy,'section':y,'frame':7,'seed':1701,'episode':str(directory),
                  'turns':gym.turn,'accepted':sum(e['accepted'] for e in gym.events),
                  'strokes':len(gym.operations)-1,'coarse_correspondence':obs['coarse_correspondence'],
                  'media':dict(Counter(o['args']['medium'] for o in gym.operations[1:])),
                  'adapter_loaded':policy=='sft','wall_seconds':time.perf_counter()-started}
            write_json(directory/'policy.json',item)
            # Correct generic gym metadata for this externally loaded trained adapter.
            manifest=json.loads((directory/'manifest.json').read_text());manifest['trained_policy']=policy=='sft'
            write_json(directory/'manifest.json',manifest)
            record['records'].append(item);write_json(out/'manifest.json',record)
            print('EPISODE',json.dumps(item),flush=True)
    record['status']='completed';write_json(out/'manifest.json',record)


if __name__=='__main__':main()
