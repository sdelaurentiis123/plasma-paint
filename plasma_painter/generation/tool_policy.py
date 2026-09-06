"""Vision-conditioned model-authored painting DSL; no artist-specific code templates."""
import json
import os
import time
from pathlib import Path
from plasma_painter.config import load_config,artifact_root,stable_hash,sha256_file,write_json
from plasma_painter.renderer.paint_program import PROGRAM_CONTRACT,compile_paint_program
from plasma_painter.generation.sample_programs import _strip_fence
from plasma_painter.generation.contract_audit import validate_candidate
from plasma_painter.renderer.render_program import render_program_clip


def main():
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from plasma_painter.features.pipeline import decode_unit_raster
    from plasma_painter.provenance import experiment_provenance

    started=time.perf_counter()
    config=load_config('configs/plasma_painter/pilot.yaml');config['renderer']['profile']='stroke_only'
    root=artifact_root(config);index=json.loads((root/'features/index.json').read_text())
    source=next(r for r in index['clips'] if r['split']=='art_train')
    clip=json.loads(Path(source['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    model_info=json.loads(Path('models/vision-model-manifest.json').read_text())
    config['generation']['base_model']=model_info['repo']
    provenance=experiment_provenance(config,wall_seconds=0,stage='model_authored_tool_policy')
    config['project']['artifact_root']=str(root/'tool-policy');out=artifact_root(config);out.mkdir(parents=True,exist_ok=False)
    refs=json.loads(Path('reference-images/manifest.json').read_text())['records']
    scalar=decode_unit_raster(clip['frames'][0]['rasters']['density_fluctuation'])
    science=Image.fromarray(np.uint8(np.clip(scalar.T,0,1)*255)).convert('RGB').resize((512,336));science.save(out/'scientific.png')
    model_path=os.environ['PLASMA_PAINTER_MODEL_PATH']
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,use_fast=False,min_pixels=256*28*28,max_pixels=512*28*28)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa');model.eval()
    records=[]
    manifest={'status':'in_progress','records':records,'model':model_info,'provenance':provenance,
              'no_training':True,'compiler':'paint-program-v1','no_artist_presets':True,'config':config}
    write_json(out/'manifest.json',manifest)
    for artist in ('Vincent van Gogh','Georges Seurat'):
        selected=[r for r in refs if r['artist_title']==artist]
        assert len(selected)==2
        images=[]
        for ref in selected:
            path=Path('reference-images')/Path(ref['file']).name
            assert ref['is_public_domain'] and sha256_file(path)==ref['sha256']
            with Image.open(path) as image:
                image=image.convert('RGB');image.thumbnail((560,560));images.append(image.copy())
        images.append(science)
        prompt=(f'Look carefully at these two {artist} paintings, then the plasma image. '
                'Choose the right painting tools, characteristic stroke shapes, scale hierarchy, '
                'colors and layered mark-making for the reference style. Do not copy depicted objects. '
                'The final image is real plasma: x right, periodic z down, negative dark, neutral gray, positive light. '
                'Make the plasma legible and the painting intentional. Write your own painting DSL program. '
                'Use the available tools freely; there is no preselected brush or palette.')
        messages=[{'role':'system','content':PROGRAM_CONTRACT},
                  {'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
        for attempt in range(3):
            seed=1701+attempt
            formatted=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[formatted],images=images,padding=True,return_tensors='pt').to(model.device)
            assert len(inputs['image_grid_thw'])==3
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=2200,do_sample=True,temperature=.65,top_p=.9)
            generated=tokens[:,inputs.input_ids.shape[1]:]
            raw=processor.batch_decode(generated,skip_special_tokens=True)[0]
            item={'artist':artist,'attempt':attempt,'generation_seed':seed,'render_seed':1701,
                  'raw_text':raw,'messages':list(messages),'reference_ids':[r['id'] for r in selected],
                  'image_grid_thw':inputs['image_grid_thw'].tolist(),'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
                  'generated_token_count':generated.shape[1],'hit_token_cap':generated.shape[1]>=2200,
                  'origin':'frozen_vision_authored_DSL_compiled_to_JS_no_painting_example'}
            try:
                program=json.loads(_strip_fence(raw));code=compile_paint_program(program)
                digest=stable_hash(program);code_hash=stable_hash(code)
                write_json(out/(digest+'.json'),program);(out/(code_hash+'.js')).write_text(code)
                item.update(program=program,dsl_hash=digest,program_hash=code_hash,
                            chosen_tools=[layer['tool'] for layer in program['layers']])
                style={'name':artist,'paper':program['paper']['color'],'grain':program['paper']['grain']}
                validity=validate_candidate(code,clip['frames'],config,style);item['validation']=validity
                if not validity['accepted']:raise ValueError(validity.get('error') or validity.get('render_error') or 'Output is blank')
                item['render']=render_program_clip(code,clip,config,style,seed=1701,checkpoint=model_info['repo'],origin=item['origin'])
            except (ValueError,RuntimeError,TypeError,KeyError) as error:
                item['error']=str(error)
            records.append(item);write_json(out/'manifest.json',manifest)
            print(f'{artist} attempt {attempt}: tools={item.get("chosen_tools")} rendered={"render" in item} error={item.get("error")}',flush=True)
            if 'render' in item:break
            messages += [{'role':'assistant','content':raw},{'role':'user','content':
                'Correct the complete JSON painting program. Compiler/runtime error: '+item['error']+
                ' Keep your intended style and choose your own tools. Return only JSON; check every required key and bound.'}]
    manifest.update(status='completed',wall_seconds=time.perf_counter()-started)
    write_json(out/'manifest.json',manifest)


if __name__=='__main__':main()
