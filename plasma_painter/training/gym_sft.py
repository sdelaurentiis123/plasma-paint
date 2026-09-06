"""Train a small vision LoRA on validated tool demonstrations, NOT artist quality."""
import argparse
import json
from pathlib import Path
import numpy as np
from plasma_painter.config import sha256_file,git_state
from plasma_painter.training.painting_gym import PaintingGym,TOOLS


def build(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cache=Path('artifacts/plasma_painter/free_sections')
    index=json.loads((cache/'index.json').read_text());rows=[]
    for section in index['records']:
        y=section['y']
        if y not in (0,18,31):raise ValueError('Unexpected section')
        path=cache/f'section-y{y}.generated.json'
        if sha256_file(path)!=section['sha256']:raise ValueError('Cache hash mismatch')
        clip=json.loads(path.read_text());assert clip['split']=='art_train'
        for offset in (0,1,6):
            frame=clip['frames'][offset];gym=PaintingGym(frame,size=(192,128))
            rng=np.random.default_rng(1701+y*10+offset)
            for turn in range(8):
                obs=gym.observe();medium=TOOLS['media'][turn%6];strokes=[]
                for _ in range(4):
                    x,z=rng.uniform(.06,.94,2)
                    v=float(gym.field[round(z*(gym.field.shape[0]-1)),round(x*(gym.field.shape[1]-1))])
                    strokes.append({'points':[[round(float(x-.025),4),round(float(z),4)],
                                              [round(float(x+.025),4),round(float(z+.01),4)]],
                        'medium':medium,'color':'#a54236' if v>.5 else '#315f87',
                        'width':.018,'opacity':round(min(.8,.1+abs(2*v-1)*.7),4)})
                action={'action':'paint','strokes':strokes}
                _,_,_,_,info=gym.step(action);assert info['accepted'],info
                stem=f'y{y}-f{offset}-t{turn}'
                images=[]
                for kind in ('scientific','canvas'):
                    img=root/f'{stem}-{kind}.png';obs[kind].save(img);images.append(img.name)
                rows.append({'id':stem,'split':'format_eval' if offset==6 else 'format_train',
                    'source':frame['source'],'section':y,'images':images,
                    'prompt':'Practice valid finite stroke tool use on this single plasma frame. First image: '
                    'scientific reference; second: current canvas. Use '+medium+'. Output one JSON paint action '
                    'with four strokes, normalized XY coordinates (not pixels), hex colors, and bounded parameters. '
                    'Tools: '+json.dumps(TOOLS),
                    'completion':json.dumps(action,separators=(',',':')),'origin':'hand_authored_tool_curriculum',
                    'aesthetic_training_eligible':False})
    (root/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'manifest.json').write_text(json.dumps({'git':git_state(),'source_index':index,
        'purpose':'action-format SFT only; not learned artist quality','train_frames':[0,1],
        'format_eval_frames':[6],'all_inside_art_train':True,'count':len(rows)},indent=2))
    return rows


def execute(dataset,model_path,output,steps):
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    import time
    if not 1<=steps<=40:raise ValueError('This smoke run permits at most 40 steps')
    out=Path(output)
    if out.exists():raise ValueError('Refuse to overwrite adapter')
    root=Path(dataset);all_rows=[json.loads(l) for l in (root/'examples.jsonl').read_text().splitlines()]
    rows=[r for r in all_rows if r['split']=='format_train']
    np.random.default_rng(1701).shuffle(rows)
    if not rows:raise ValueError('No training demonstrations')
    assert all(str(r['source']['shot'])=='85604' and r['source']['frame_index'] in (0,1) for r in rows)
    torch.manual_seed(1701)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,
        min_pixels=128*28*28,max_pixels=128*28*28)
    base=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa')
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(base,config);model.train();model.config.use_cache=False
    model.gradient_checkpointing_enable();model.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    def format_eval():
        results=[];model.eval()
        from plasma_painter.renderer.dsl import validate_operation
        eval_rows=[next(r for r in all_rows if r['split']=='format_eval' and r['section']==y) for y in (0,18,31)]
        for row in eval_rows:
            images=[]
            for name in row['images']:
                with Image.open(root/name) as im:images.append(im.convert('RGB'))
            message={'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':row['prompt']}]}
            prompt=processor.apply_chat_template([message],tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[prompt],images=images,return_tensors='pt').to(model.device)
            with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=700,do_sample=False)
            raw=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            error=None
            try:
                action=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
                if not isinstance(action,dict) or set(action)!={'action','strokes'} or action['action']!='paint':raise ValueError('Expected paint action')
                if not isinstance(action['strokes'],list) or len(action['strokes'])!=4:raise ValueError('Expected four strokes')
                for stroke in action['strokes']:
                    if set(stroke)-{'points','medium','color','width','opacity','pressure','texture'}:raise ValueError('Unknown stroke keys')
                    validate_operation({'op':'paintStroke','args':stroke},max_path_points=64)
            except (ValueError,TypeError,KeyError) as exc:error=str(exc)
            results.append({'id':row['id'],'raw':raw,'valid':error is None,'error':error})
        model.train();return results
    losses=[];started=time.perf_counter();before=format_eval()
    print('FORMAT EVAL BEFORE',json.dumps(before),flush=True)
    for step in range(steps):
        row=rows[step%len(rows)];images=[]
        for name in row['images']:
            if Path(name).name!=name:raise ValueError('Invalid image path')
            with Image.open(root/name) as im:images.append(im.convert('RGB'))
        user={'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':row['prompt']}]}
        prefix=processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
        full=processor.apply_chat_template([user,{'role':'assistant','content':row['completion']}],tokenize=False)
        inputs=processor(text=[full],images=images,return_tensors='pt').to(model.device)
        prompt=processor(text=[prefix],images=images,return_tensors='pt').input_ids.to(model.device)
        n=prompt.shape[1]
        if not torch.equal(inputs.input_ids[:,:n],prompt):raise ValueError('Prompt token prefix mismatch; refusing wrong loss mask')
        labels=inputs.input_ids.clone();labels[:,:n]=-100
        if not (labels!=-100).any():raise ValueError('Empty completion labels')
        loss=model(**inputs,labels=labels).loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite SFT loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()));print('GYM SFT',step+1,losses[-1],flush=True)
    after=format_eval()
    print('FORMAT EVAL AFTER',json.dumps(after),flush=True)
    out.mkdir(parents=True,exist_ok=False);model.save_pretrained(out);processor.save_pretrained(out)
    (out/'run.json').write_text(json.dumps({'steps':steps,'losses':losses,'model':model_path,
        'dataset_sha256':sha256_file(root/'examples.jsonl'),'git':git_state(),'seed':1701,
        'wall_seconds':time.perf_counter()-started,'hardware':torch.cuda.get_device_name(),
        'purpose':'valid-tool-use SFT, not aesthetic or RL training','adapter':config.to_dict(),
        'format_eval_before':before,'format_eval_after':after},indent=2,default=list))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--build',action='store_true')
    p.add_argument('--model-path');p.add_argument('--output');p.add_argument('--steps',type=int,default=32)
    a=p.parse_args()
    if a.build:print('examples',len(build(a.dataset)))
    if a.model_path:
        if not a.output:p.error('--output required for training')
        execute(a.dataset,a.model_path,a.output,a.steps)


if __name__=='__main__':main()
