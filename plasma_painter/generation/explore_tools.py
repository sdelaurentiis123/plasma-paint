"""Frozen-model tool exploration: briefs, no renderer examples; one validity repair."""
import json
import os
import time
from pathlib import Path
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from plasma_painter.config import load_config,artifact_root,stable_hash,write_json
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.generation.sample_programs import _strip_fence
from plasma_painter.generation.filter_programs import filter_candidate
from plasma_painter.renderer.render_program import render_program_clip
from plasma_painter.provenance import experiment_provenance

BRIEFS={
    'expressive':'An expressive Post-Impressionist painting inspired by Van Gogh. Invent your own composition and mark-making strategy for the data.',
    'pointillist':'A luminous Neo-Impressionist painting inspired by Seurat. Choose your own tools and color relationships.',
    'cubist':'An analytical Cubist interpretation inspired by Picasso. Preserve scientific spatial structure while developing your own visual vocabulary.',
    'tonal':'A painterly tonal interpretation inspired by Manet. Choose what to emphasize and how to build the painting.',
    'drawing':'A compelling pencil or charcoal drawing. Develop your own drawing technique and hierarchy of marks.',
    'pop':'A bold Pop Art interpretation inspired by Warhol. Choose your own palette and mark-making, without duplicating or moving plasma structures.',
}


def prompt_for(brief):
    return '''Write an executable reusable painting program, not an image description.
Return ONLY JavaScript, beginning exactly export function createPainter(api, styleConfig).
Return an object with reset(seed) and renderFrame(frameFeatures,time,persistentState).
The fixed renderer offers generic tools; YOU choose mark placement, number, order, pressure, medium, color and layering. No artist-specific renderer exists. No reference program is provided.
Available calls (one object argument except reset):
api.reset(seed), used from reset(seed).
api.createPaper({color: six-digit hex string, grain: number 0..0.02}), exactly once per frame.
api.mark({points: array of 2..8 normalized [x,z] points, width: .0005..0.018,
opacity: 0..0.8, pressure: .05..1, texture: 0..1,
medium: 'bristle'|'graphite'|'charcoal'|'ink'|'pastel'|'watercolor',
color: six-digit hex string, sample_id: integer}). All calls return undefined.
Each mark must have total polyline length .002..0.06. All coordinates must be in [0,1].
At most 1100 marks per frame. Build tone with finite marks; no raster wash, whole contour, bloom, or other drawing API.
frameFeatures.stroke_samples is an array of {id,x,z,value,tx,tz}. value is signed density fluctuation encoded [0,1], zero at .5, fitted on training data. tx,tz are display-tangent visualization proxies, NOT physical flow. Anchors sample the permitted field. Every point of a mark must be within .04 of its sample_id anchor. You may select anchors, offset marks, reuse anchors and choose different numbers of marks in different regions; you are NOT required to draw a grid or one mark per sample. Maintain meaningful spatial and signed-intensity correspondence, and respond to every new frame. Do not relocate or invent plasma structures.
You can compute and mix your own RGB colors, converting them to six-digit hex strings. No fixed palette or medium is prescribed by the runtime.
Use bounded loops and basic Math functions. No DOM, network, imports, dynamic code, direct Canvas, p5, timers, while loops, classes, or host globals. Call api.mark directly, without aliasing. Do not output prose or Markdown fences.
Style brief: '''+brief


def main():
    config=load_config('configs/plasma_painter/pilot.yaml')
    config['renderer']['profile']='stroke_only'
    root=artifact_root(config)
    config['project']['artifact_root']=str(root/'tool-exploration')
    out=artifact_root(config);out.mkdir(parents=True,exist_ok=True)
    index=json.loads((root/'features/index.json').read_text())
    record=next(r for r in index['clips'] if r['split']=='art_train')
    clip=json.loads(Path(record['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    frames=[with_stroke_samples(f) for f in clip['frames']]
    model_path=os.environ['PLASMA_PAINTER_MODEL_PATH']
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype='auto',device_map='auto');model.eval()
    started=time.perf_counter();records=[]
    for name,brief in BRIEFS.items():
        prompt=prompt_for(brief)
        style={'name':name,'grain':0,'paper':'#f7f0df'}
        messages=[{'role':'user','content':prompt}]
        for attempt in range(2):
            seed=1701+attempt;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            ids=tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated=model.generate(ids,attention_mask=torch.ones_like(ids),pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=2600,do_sample=True,temperature=.8,top_p=.95)[0]
            raw=tokenizer.decode(generated[ids.shape[1]:],skip_special_tokens=True)
            code=_strip_fence(raw).rstrip()+'\n';digest=stable_hash(code)
            path=out/(digest+'.js');path.write_text(code)
            status=filter_candidate(code,frames,config,style)
            item={'style_brief':brief,'style':name,'attempt':attempt,'seed':seed,'origin':'frozen_base_no_renderer_example',
                  'program_hash':digest,'program_path':str(path),'prompt_hash':stable_hash(messages),'raw_text':raw,'messages':messages.copy(),'validation':status}
            if status['accepted']:
                try:
                    item['render']=render_program_clip(code,clip,config,style,seed=seed,checkpoint=config['generation']['base_model'],origin=item['origin'])
                except (RuntimeError,ValueError,KeyError,TypeError) as error:
                    item['render_error']=str(error)
            records.append(item);write_json(out/'manifest.json',{'status':'in_progress','records':records})
            print(f'{name} attempt {attempt}: valid={status["accepted"]}, rendered={"render" in item}',flush=True)
            if 'render' in item: break
            messages += [{'role':'assistant','content':raw},{'role':'user','content':'Repair this program using the same tool contract. Validator feedback: '+str(status.get('error') or item.get('render_error') or 'no visible pigment; produce nonempty marks')+' Return only complete JavaScript.'}]
    write_json(out/'manifest.json',{'status':'completed','records':records,'provenance':experiment_provenance(config,wall_seconds=time.perf_counter()-started,stage='frozen_tool_exploration'),'feature_file':record['path'],'clip':clip['clip_id'],'no_aesthetic_judge':True,'no_optimizer_updates':True})


if __name__=='__main__':main()
