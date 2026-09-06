"""Run a supplied stroke demonstration or local vision policy in the single-frame gym."""
import argparse
import json
from pathlib import Path
import numpy as np
from plasma_painter.features.pipeline import encode_unit_raster
from plasma_painter.training.painting_gym import PaintingGym


def demonstration(env):
    """Auditable tool-use curriculum, NOT a learned artist or style recipe."""
    rng=np.random.default_rng(env.seed)
    for batch in range(min(8,env.max_turns-1)):
        strokes=[]
        for i in range(16):
            x,y=rng.uniform(.04,.96,2)
            v=float(env.field[round(y*(env.field.shape[0]-1)),round(x*(env.field.shape[1]-1))])
            strokes.append({'points':[[float(x-.025),float(y-.015)],[float(x+.025),float(y+.015)]],
                'medium':('watercolor','bristle','graphite','charcoal','ink','pastel')[batch%6],
                'color':'#b34d3d' if v>.5 else '#305e83','width':.04,
                'opacity':min(.8,.05+abs(2*v-1)*.75)})
        env.step({'action':'paint','strokes':strokes})
    env.step({'action':'finish'})


def vision_rollout(env, model_path, references):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,
        min_pixels=256*28*28,max_pixels=512*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa')
    model.eval();torch.manual_seed(env.seed)
    refs=[]
    for path in references:
        with Image.open(path) as im:refs.append(im.convert('RGB'))
    previous=None
    while not env.done:
        obs=env.observe();images=refs+[obs['scientific'],obs['canvas']]
        prompt=('Paint the plasma in the penultimate image onto the last image (your current canvas). '
            'Earlier images are style references, not objects to copy. Choose tools, colors and free XY '
            'stroke paths. Output ONE JSON action only, no code. Work incrementally: at most 4 strokes '
            'in this response. Inspect the updated canvas on the next turn. '
            'X is radial, Y is field-aligned z; this is one fixed frame, no time. '
            'Tools: '+json.dumps(obs['tools'])+' Valid syntax example (tool mechanics only, not an artist recipe): '
            '{"action":"paint","strokes":[{"points":[[0.2,0.3],[0.25,0.35]],"medium":"ink",'
            '"color":"#254060","width":0.01,"opacity":0.5}]} '
            'Other actions: {"action":"undo"}, {"action":"finish"}. '
            f'Turns left {obs["turns_left"]}; strokes left {obs["strokes_left"]}. '
            'Last action feedback: '+json.dumps(previous))
        messages=[{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[text],images=images,return_tensors='pt').to(model.device)
        with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=700,do_sample=True,temperature=.7)
        raw=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
        try:action=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
        except ValueError:action={'action':'invalid_json'}
        _,_,_,_,previous=env.step(action)
        env.events[-1]['model_raw_response']=raw
        print(env.turn,previous,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--section',type=int,choices=[0,18,31],default=18)
    p.add_argument('--frame-offset',type=int,choices=range(8),default=0)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--model-path',help='Existing local vision model; no downloading or external API')
    p.add_argument('--reference',action='append',default=[],help='Explicitly permitted local reference image (max 2)')
    args=p.parse_args()
    if len(args.reference)>2:p.error('At most two references')
    if Path(args.output).exists():p.error('Output already exists; choose a new episode directory')
    if args.synthetic:
        x,y=np.mgrid[0:1:32j,0:1:48j]
        values=.5+.45*np.sin(2*np.pi*y)*np.exp(-((x-.5)/.23)**2)
        frame={'rasters':{'density_fluctuation':encode_unit_raster(values)},'source':{'label':'SYNTHETIC'},'geometry':{}}
    else:
        root=Path('artifacts/plasma_painter/free_sections')
        index=json.loads((root/'index.json').read_text())
        record=next(r for r in index['records'] if r['y']==args.section)
        from plasma_painter.config import sha256_file
        path=root/f'section-y{args.section}.generated.json'
        if sha256_file(path)!=record['sha256']:raise ValueError('Staged section hash mismatch')
        clip=json.loads(path.read_text())
        if clip['split']!='art_train':raise ValueError('Only art_train is allowed')
        frame=clip['frames'][args.frame_offset]
    env=PaintingGym(frame,synthetic=args.synthetic)
    if args.model_path:vision_rollout(env,args.model_path,args.reference)
    else:demonstration(env)
    out=env.save(args.output)
    (out/'run.json').write_text(json.dumps({'policy':'frozen_local_vision' if args.model_path else 'hand_authored_tool_demonstration',
        'model':args.model_path,'references':args.reference,'training_performed':False,'synthetic':args.synthetic},indent=2))
    print(out)


if __name__=='__main__':main()
