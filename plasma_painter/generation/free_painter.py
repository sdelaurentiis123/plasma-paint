"""Free-XY code generation with section transfer and image-space diagnostics."""
import json
import os
import time
from pathlib import Path
from plasma_painter.config import load_config,artifact_root,stable_hash,sha256_file,write_json
from plasma_painter.generation.free_paint_prompt import CONTRACT,wrap_body,prepare_response
from plasma_painter.generation.sample_programs import _strip_fence
from plasma_painter.generation.contract_audit import validate_candidate
from plasma_painter.renderer.render_program import render_program_clip
from plasma_painter.evaluation.finite_probe import probe


def main():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from plasma_painter.features.pipeline import decode_unit_raster
    from plasma_painter.provenance import experiment_provenance
    started=time.perf_counter()
    config=load_config('configs/plasma_painter/pilot.yaml');config['renderer']['profile']='free_paint'
    root=artifact_root(config);section_root=root/'free_sections'
    index=json.loads((section_root/'index.json').read_text());assert index['status']=='complete'
    clips=[];science=[]
    for record in index['records']:
        path=Path(record['path'])
        if path.is_absolute() or len(path.parts)!=1:raise ValueError('Invalid staged clip path')
        path=section_root/path;assert sha256_file(path)==record['sha256']
        clip=json.loads(path.read_text())
        assert clip['split']=='art_train' and all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
        clips.append(clip)
        values=decode_unit_raster(clip['frames'][0]['rasters']['density_fluctuation'])
        science.append(Image.fromarray(np.uint8(np.clip(values.T,0,1)*255)).convert('RGB').resize((384,256)))
    assert [r['y'] for r in index['records']]==[0,18,31]
    model_info=json.loads(Path('models/vision-model-manifest.json').read_text())
    config['generation']['base_model']=model_info['repo']
    provenance=experiment_provenance(config,wall_seconds=0,stage='free_xy_generation')
    config['project']['artifact_root']=str(root/'free-paint');out=artifact_root(config);out.mkdir(parents=True,exist_ok=False)
    refs=json.loads(Path('reference-images/manifest.json').read_text())['records']
    model_path=os.environ['PLASMA_PAINTER_MODEL_PATH']
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,min_pixels=256*28*28,max_pixels=512*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa');model.eval()
    records=[]
    manifest={'status':'in_progress','records':records,'model':model_info,'provenance':provenance,
              'sections':index,'no_training':True,'profile':'free_paint','config':config,
              'scientific_scope':'partial coarse-intensity and response floors; full fidelity and human preference still required'}
    write_json(out/'manifest.json',manifest)
    for artist in ('Vincent van Gogh','Georges Seurat'):
        selected=[r for r in refs if r['artist_title']==artist];assert len(selected)==2
        images=[]
        for ref in selected:
            path=Path('reference-images')/Path(ref['file']).name
            assert ref['is_public_domain'] and sha256_file(path)==ref['sha256']
            with Image.open(path) as image:
                image=image.convert('RGB');image.thumbnail((560,560));images.append(image.copy())
        images+=science
        prompt=(f'Use the first two images ({artist} paintings) to choose your own tools and mark-making. '
                'The next three images are real plasma density-fluctuation cross-sections y=0,18,31. '
                'Paint these evolving structures, not the objects in the artworks. '
                'You may paint anywhere across the full canvas. Design a reusable algorithm that samples '
                'the provided field and adapts its stroke placement, path shape, pigment and tools. '
                'Do not branch on section IDs or hard-code a particular training frame. '
                'Preserve strong structures and signed intensity. Build a coherent, richly painted image, '
                'not sparse grid stamps or data-independent random decoration. '
                'Choose your own palette, layering and spatial organization. Return one complete '
                'function renderFrame(frameFeatures,time,persistentState). Check that all coordinates '
                'are declared before sampling, gradient returns .dx and .dz (not an array), colors '
                'are six-digit hex, every path has at least two points, and field names are exact.')
        messages=[{'role':'system','content':CONTRACT},
                  {'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
        for attempt in range(3):
            seed=1701+attempt
            formatted=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[formatted],images=images,padding=True,return_tensors='pt').to(model.device)
            assert len(inputs['image_grid_thw'])==len(images)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=2800,do_sample=True,temperature=.65,top_p=.9)
            generated=tokens[:,inputs.input_ids.shape[1]:]
            raw=processor.batch_decode(generated,skip_special_tokens=True)[0]
            preparation_error=None
            try:code,response_format=prepare_response(raw)
            except ValueError as error:
                preparation_error=str(error);code=wrap_body(_strip_fence(raw));response_format='invalid'
            digest=stable_hash(code)
            (out/(digest+'.js')).write_text(code)
            item={'artist':artist,'attempt':attempt,'generation_seed':seed,'render_seed':1701,'raw_text':raw,
                  'messages':list(messages),'reference_ids':[r['id'] for r in selected],
                  'program_hash':digest,'program_path':str(out/(digest+'.js')),
                  'response_format':response_format,
                  'duplicate_of_attempt':next((r['attempt'] for r in records if r['artist']==artist and r['program_hash']==digest),None),
                  'input_shapes':{k:list(v.shape) for k,v in inputs.items()},'image_grid_thw':inputs['image_grid_thw'].tolist(),
                  'generated_token_count':generated.shape[1],'hit_token_cap':generated.shape[1]>=2800,
                  'origin':'frozen_vision_free_xy_body_trusted_lifecycle_no_painting_example','sections':[]}
            try:
                if preparation_error:raise ValueError(preparation_error)
                if item['duplicate_of_attempt'] is not None:
                    previous=next(r for r in records if r['artist']==artist and r['program_hash']==digest)
                    raise ValueError('Unchanged rejected program; repair the actual error: '+previous.get('error','invalid output'))
                for clip,section in zip(clips,index['records']):
                    style={'name':artist,'grain':0,'paper':'#f7f0df'}
                    valid=validate_candidate(code,clip['frames'],config,style)
                    detail={'y':section['y'],'validation':valid};item['sections'].append(detail)
                    if not valid['accepted']:raise ValueError(valid.get('error') or valid.get('render_error') or 'blank output')
                    detail['probe']=probe(code,clip,1701,profile='free_paint')
                    detail['render']=render_program_clip(code,clip,config,style,seed=1701,checkpoint=model_info['repo'],origin=item['origin'])
                    if section['y']==18:item['render']=detail['render']
                    p=detail['probe']
                    detail['partial_science_checks']={
                        'coarse_floor':p['coarse_abs_fluctuation_pigment_spearman']>=section['coarse_floor'],
                        'reproducible':p['pixels_reproducible'] and p['operations_reproducible'],
                        'data_response':p['evolving_input_mean_rgb_change']>p.get('frozen_input_mean_rgb_change',1)+.0001,
                        'frozen_input_stable':p.get('frozen_input_mean_rgb_change',1)<.00001}
                item['partial_science_pass']=all(all(s['partial_science_checks'].values()) for s in item['sections'])
                if not item['partial_science_pass']:
                    item['error']='Image-space checks failed: '+json.dumps([{ 'section':s['y'],'checks':s['partial_science_checks'],
                        'correlation':s['probe']['coarse_abs_fluctuation_pigment_spearman'],
                        'required':index['records'][i]['coarse_floor']} for i,s in enumerate(item['sections'])])
            except (ValueError,RuntimeError,TypeError,KeyError) as error:item['error']=str(error)
            records.append(item);write_json(out/'manifest.json',manifest)
            print(f'{artist} attempt {attempt}: rendered_sections={sum("render" in s for s in item["sections"])} partial_science={item.get("partial_science_pass",False)} error={item.get("error")}',flush=True)
            if item.get('partial_science_pass'):break
            feedback='Correct the reusable renderFrame function. Evidence: '+item.get('error','incomplete render')+'. Keep the intended artist style and free stroke placement; preserve the plasma. Return the complete corrected function, not the identical rejected program.'
            if 'render' in item:
                with Image.open(item['render']['still']) as image:
                    image=image.convert('RGB');image.thumbnail((560,560));images.append(image.copy())
                user_content=[{'type':'image'},{'type':'text','text':'This last image is your current painting. '+feedback}]
            else:user_content=feedback
            messages += [{'role':'assistant','content':raw},{'role':'user','content':user_content}]
    manifest.update(status='completed',wall_seconds=time.perf_counter()-started)
    write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
