"""Bounded diagnostic ablation, not training or an aesthetic benchmark."""
from __future__ import annotations

import gc
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from plasma_painter.config import artifact_root, load_config, stable_hash, write_json, sha256_file
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.generation.filter_programs import filter_candidate
from plasma_painter.generation.sample_programs import _strip_fence
from plasma_painter.generation.tool_contract import INTERFACE_GUIDANCE
from plasma_painter.renderer.render_program import render_program_clip


CASES = (
    ('vl_text_scratch', 'vision', False, False),
    ('vl_image_scratch', 'vision', True, False),
    ('vl_image_example', 'vision', True, True),
    ('coder_text_scratch', 'coder', False, False),
)


def make_prompt(frame, reference, *, example=False):
    samples = with_stroke_samples(frame)['stroke_samples']
    brief = '''Begin literally: export function createPainter(api, styleConfig) {
Write a complete reusable JavaScript program, with reset(seed) { api.reset(seed); }.
Aim for expressive directional finite brushstrokes and purposeful color relationships,
inspired by Van Gogh. Depict the plasma, not objects from reference paintings.
There is no artist-specific API. Choose your own marks, media, palette and layering.
Return reset(seed) and renderFrame(frameFeatures,time,persistentState).
Only api.reset, api.createPaper and api.mark are allowed. At most 1100 marks/frame.
mark bounds: 2..8 [x,z] points, total path length .002.. .06, width .0005.. .018,
opacity 0.. .8, pressure .05..1, texture 0..1. Medium is bristle, graphite, charcoal,
ink, pastel or watercolor. Colors must be #rrggbb. Drawing calls return undefined.
No imports, DOM, image loading, direct Canvas, async, timers, while or do loops.
Use bounded for loops over frameFeatures.stroke_samples. Every major mark must
retain meaningful position and signed intensity from its referenced sample.
The array below shows actual input shape, NOT the complete frame. Render the full
array supplied on every call. tx,tz are visualization tangents, NOT physical flow.
'''
    brief += '\nRuntime field subset (array truncated for illustration): ' + json.dumps({
        'stroke_samples': samples[:3],
    }, sort_keys=True)
    brief += (f'\nExplanatory metadata, NOT frameFeatures properties: sample_count={len(samples)}, '
              'value_range=[0,1], neutral_value=0.5. Read the count from '
              'frameFeatures.stroke_samples.length. Never access frameFeatures.value_range, '
              'frameFeatures.neutral_value or frameFeatures.sample_count.')
    if example:
        brief += ('\nWORKING INTERFACE CONTROL, hand-authored, not artist training data. '
                  'Redesign its mark-making while preserving the exact interface and bounds.\n' + reference)
    return brief


def validate_candidate(code, frames, config, style):
    """A cold Node startup is an infrastructure retry, never a model repair."""
    result = filter_candidate(code, frames, config, style)
    first = result
    if result.get('error') == 'JavaScript syntax check timed out after 3 seconds':
        subprocess.run(['node', '-e', ''], check=True, timeout=30, capture_output=True)
        result = filter_candidate(code, frames, config, style)
        result['infrastructure_retry'] = {'reason': first['error'], 'same_program': True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Validate controls and exact token/image inputs without loading model weights or using a GPU')
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
    from plasma_painter.features.pipeline import decode_unit_raster
    from plasma_painter.provenance import experiment_provenance

    started = time.perf_counter()
    config = load_config('configs/plasma_painter/pilot.yaml')
    config['renderer']['profile'] = 'stroke_only'
    config['project']['seed'] = 1701
    root = artifact_root(config)
    index = json.loads((root/'features/index.json').read_text())
    source = next(r for r in index['clips'] if r['split'] == 'art_train')
    clip = json.loads(Path(source['path']).read_text())
    assert all(str(f['source']['shot']) == '85604' and 0 <= f['source']['frame_index'] < 288 for f in clip['frames'])
    provenance = experiment_provenance(config, wall_seconds=0, stage='contract_ablation_inputs')
    config['project']['artifact_root'] = str(root/('contract-preflight' if args.preflight else 'contract-audit'))
    out = artifact_root(config)
    out.mkdir(parents=True, exist_ok=False)
    reference = Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    style = {'name': 'contract-ablation', 'grain': 0, 'paper': '#f7f0df'}
    subprocess.run(['node', '-e', ''], check=True, timeout=30, capture_output=True)
    control = validate_candidate(reference, clip['frames'], config, style)
    if not control['accepted']:
        write_json(out/'manifest.json', {'status': 'preflight_failed', 'control': control})
        raise RuntimeError('Known working control failed; refusing model sampling')
    refs = [r for r in json.loads(Path('reference-images/manifest.json').read_text())['records']
            if r['artist_title'] == 'Vincent van Gogh']
    assert len(refs) == 2
    images = []
    for ref in refs:
        path = Path('reference-images')/Path(ref['file']).name
        assert ref['is_public_domain'] and sha256_file(path) == ref['sha256']
        with Image.open(path) as original:
            image = original.convert('RGB'); image.thumbnail((560, 560)); images.append(image.copy())
    scalar = decode_unit_raster(clip['frames'][0]['rasters']['density_fluctuation'])
    science = Image.fromarray(np.uint8(np.clip(scalar.T, 0, 1)*255)).convert('RGB').resize((512, 336))
    science.save(out/'scientific-frame-0.png'); images.append(science)
    records = []
    manifest = {'status': 'in_progress', 'records': records, 'control': control,
                'input_provenance': provenance, 'reference_records': refs,
                'vision_model_manifest': json.loads(Path('models/vision-model-manifest.json').read_text()),
                'feature_sha256': sha256_file(source['path']), 'no_training': True,
                'adapter_loaded': False, 'not_a_fidelity_or_aesthetic_benchmark': True}
    write_json(out/'manifest.json', manifest)
    for family in ('vision', 'coder'):
        path = os.environ['PLASMA_PAINTER_MODEL_PATH' if family == 'vision' else 'PLASMA_PAINTER_CODER_PATH']
        if family == 'vision':
            processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False,
                                                      min_pixels=256*28*28, max_pixels=512*28*28)
            cls = Qwen2_5_VLForConditionalGeneration
        else:
            processor = AutoTokenizer.from_pretrained(path, local_files_only=True)
            cls = AutoModelForCausalLM
        model = None
        if not args.preflight:
            model = cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                        device_map='auto', attn_implementation='sdpa'); model.eval()
        for case, model_family, use_images, example in CASES:
            if model_family != family: continue
            prompt = make_prompt(clip['frames'][0], reference, example=example)
            if use_images:
                prompt += '\nImages: two artist references, then plasma (x right, periodic z down; negative dark, neutral gray, positive light).'
            content = ([{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]) if use_images else prompt
            messages = [{'role': 'system', 'content': INTERFACE_GUIDANCE}, {'role': 'user', 'content': content}]
            formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            kwargs = {'images': images} if use_images else {}
            inputs = processor(text=[formatted], padding=True, return_tensors='pt', **kwargs)
            assert formatted.endswith('<|im_start|>assistant\n')
            assert INTERFACE_GUIDANCE in formatted and prompt in formatted
            assert formatted.count('<|image_pad|>') == (len(images) if use_images else 0)
            if use_images:
                assert len(inputs['image_grid_thw']) == len(images)
            if args.preflight:
                records.append({'case':case, 'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
                                'formatted_prompt':formatted, 'image_count':len(images) if use_images else 0})
                write_json(out/'manifest.json',manifest)
                print(f'{case}: exact chat/image preflight passed',flush=True)
                continue
            inputs = inputs.to(model.device)
            generation = {'max_new_tokens': 2800, 'do_sample': True, 'temperature': .7, 'top_p': .9,
                          'repetition_penalty': 1.05}
            torch.manual_seed(1701); torch.cuda.manual_seed_all(1701)
            call_started = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(**inputs, **generation)
            generated = tokens[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            code = _strip_fence(raw).rstrip()+'\n'; digest = stable_hash(code)
            (out/(digest+'.js')).write_text(code)
            validity = validate_candidate(code, clip['frames'], config, style)
            item = {'case': case, 'model_family': family, 'model_path': path, 'seed': 1701,
                    'model_id': 'Qwen/Qwen2.5-VL-7B-Instruct' if family == 'vision' else 'Qwen/Qwen2.5-Coder-7B-Instruct',
                    'model_config_sha256': sha256_file(Path(path)/'config.json'),
                    'messages': messages, 'formatted_prompt': formatted,
                    'input_shapes': {k:list(v.shape) for k,v in inputs.items()},
                    'image_count': len(images) if use_images else 0,
                    'image_grid_thw': inputs['image_grid_thw'].tolist() if use_images else None,
                    'generation_overrides': generation, 'model_generation_config': model.generation_config.to_dict(),
                    'generated_token_count': generated.shape[1], 'last_token_id': int(generated[0,-1]),
                    'hit_token_cap': generated.shape[1] >= generation['max_new_tokens'],
                    'raw_text': raw, 'program_hash': digest, 'validation': validity,
                    'seconds': time.perf_counter()-call_started,
                    'origin': 'frozen_model_with_example' if example else 'frozen_model_from_scratch'}
            if validity['accepted']:
                item['render'] = render_program_clip(code, clip, config, style, seed=1701,
                                                     checkpoint=path, origin=item['origin'])
            records.append(item); write_json(out/'manifest.json', manifest)
            print(f'{case}: accepted={validity["accepted"]}, token_cap={item["hit_token_cap"]}', flush=True)
            del inputs, tokens, generated
        del model, processor
        gc.collect(); torch.cuda.empty_cache()
    manifest.update(status='completed', wall_seconds=time.perf_counter()-started)
    write_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
