"""Image-conditioned drafts and visual revisions; no optimizer or external inference."""
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from plasma_painter.config import load_config,artifact_root,stable_hash,write_json,sha256_file
from plasma_painter.features.pipeline import decode_unit_raster
from plasma_painter.generation.explore_tools import prompt_for
from plasma_painter.generation.sample_programs import _strip_fence
from plasma_painter.generation.filter_programs import filter_candidate
from plasma_painter.renderer.render_program import render_program_clip
from plasma_painter.provenance import experiment_provenance


def main():
    started=time.perf_counter();config=load_config('configs/plasma_painter/pilot.yaml');config['renderer']['profile']='stroke_only'
    root=artifact_root(config);index=json.loads((root/'features/index.json').read_text())
    source=next(x for x in index['clips'] if x['split']=='art_train');clip=json.loads(Path(source['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    model_info=json.loads(Path('models/vision-model-manifest.json').read_text())
    config['generation']['base_model']=model_info['repo']
    input_provenance=experiment_provenance(config,wall_seconds=0,stage='vision_context_inputs')
    config['project']['artifact_root']=str(root/'vision-context');out=artifact_root(config);out.mkdir(parents=True,exist_ok=True)
    references=json.loads(Path('reference-images/manifest.json').read_text())['records']
    for ref in references:
        assert ref['is_public_domain'] is True
        assert sha256_file(Path('reference-images')/Path(ref['file']).name)==ref['sha256']
    scalar=decode_unit_raster(clip['frames'][0]['rasters']['density_fluctuation'])
    science=Image.fromarray(np.uint8(np.clip(scalar.T,0,1)*255)).convert('RGB').resize((512,336))
    science.save(out/'scientific-frame-0.png')
    model_path=os.environ['PLASMA_PAINTER_MODEL_PATH']
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,min_pixels=256*28*28,max_pixels=512*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa');model.eval()
    results=[]
    for artist in ('Vincent van Gogh','Georges Seurat','Édouard Manet'):
        refs=[r for r in references if r['artist_title']==artist]
        images=[]
        for ref in refs:
            with Image.open(Path('reference-images')/Path(ref['file']).name) as image:
                image=image.convert('RGB');image.thumbnail((560,560));images.append(image.copy())
        images.append(science)
        initial=prompt_for(f'Learn visual structure, mark-making and color relationships from the attached {artist} references. Paint the supplied plasma, not the objects in those paintings.')
        initial+='\nImages in order: '+', '.join(r['title'] for r in refs)+', followed by the scientific density-fluctuation frame (grayscale: negative dark, zero middle gray, positive light). x radial increases right; z periodic increases down. Use the scientific image to understand structures, but write a renderer that consumes arbitrary frameFeatures rather than copying this image. No renderer example is supplied.'
        text=initial
        for attempt in range(2):
            content=[{'type':'image'} for _ in images]+[{'type':'text','text':text}]
            chat=[{'role':'user','content':content}]
            formatted=processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[formatted],images=images,padding=True,return_tensors='pt').to(model.device)
            torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
            with torch.inference_mode(): tokens=model.generate(**inputs,max_new_tokens=2800,do_sample=True,temperature=.7,top_p=.9)
            raw=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
            code=_strip_fence(raw).rstrip()+'\n';digest=stable_hash(code);(out/(digest+'.js')).write_text(code)
            style={'name':artist,'grain':0,'paper':'#f7f0df'}
            validity=filter_candidate(code,clip['frames'],config,style)
            item={'artist':artist,'attempt':attempt,'seed':1701,'raw_text':raw,'program_hash':digest,'program_path':str(out/(digest+'.js')),
                  'prompt':text,'image_count':len(images),'reference_ids':[r['id'] for r in refs],'validation':validity,'origin':'frozen_vision_model_no_template'}
            if validity['accepted']:
                try:item['render']=render_program_clip(code,clip,config,style,seed=1701,checkpoint=model_info['repo'],origin=item['origin'])
                except (ValueError,RuntimeError,TypeError,KeyError) as error:item['render_error']=str(error)
            results.append(item)
            write_json(out/'manifest.json',{'status':'in_progress','model':model_info,'records':results,'input_provenance':input_provenance})
            print(f'{artist} attempt {attempt}: valid={validity["accepted"]} rendered={"render" in item}',flush=True)
            if attempt==0:
                if 'render' in item:
                    with Image.open(item['render']['still']) as image:
                        image=image.convert('RGB');image.thumbnail((560,560));images.append(image.copy())
                    feedback='The LAST image is your own rendered draft. Compare it against the artist references and scientific structure. Improve the mark organization, coherence, color and style while preserving the plasma. Return a revised complete program.'
                else:feedback='Your draft failed validation/rendering: '+str(validity.get('error') or item.get('render_error') or 'blank output')+'. Repair the complete program.'
                text=initial+'\nPrevious program:\n'+code+'\n'+feedback
    write_json(out/'manifest.json',{'status':'completed','model':model_info,'records':results,'input_provenance':input_provenance,
                                  'wall_seconds':time.perf_counter()-started,'no_training':True,'no_calibrated_aesthetic_judge':True})


if __name__=='__main__':main()
