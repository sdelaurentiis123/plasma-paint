"""Build separately typed style references and scientific frame targets, locally."""
import argparse
import json
import shutil
from pathlib import Path
from plasma_painter.config import sha256_file,stable_hash,git_state,write_json
from plasma_painter.training.painting_gym import PaintingGym


def resolve_style(pool_path,style):
    path=Path(pool_path);pool=json.loads(path.read_text())
    if pool['version']!='paired-style-targets-v1':raise ValueError('Unknown reference pool version')
    selected=[r for r in pool['references'] if r['style']==style and r['tier']=='love']
    if not selected:raise ValueError('No approved style references')
    paths=[]
    for ref in selected:
        relative=Path(ref['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe reference path')
        image=path.parent/relative
        if ref['role']!='style_reference' or ref['is_public_domain'] is not True or sha256_file(image)!=ref['sha256']:
            raise ValueError('Reference provenance/hash failed')
        paths.append(str(image))
    return paths


def build(output):
    out=Path(output)
    if out.exists():raise ValueError('Refuse to overwrite reference pool')
    config=json.loads(Path('configs/plasma_painter/style_reference_pool.json').read_text())
    library_path=Path('data/art_references/aic/manifest.json')
    library=json.loads(library_path.read_text())['records']
    references=[]
    for name,style in config['styles'].items():
        for tier in ('love','okay'):
            for identifier in style[tier]:
                original=next(r for r in library if r['id']==identifier)
                if original['artist_title']!=style['artist'] or original['is_public_domain'] is not True:
                    raise ValueError('Unexpected artist or missing reuse evidence')
                source=Path(original['file'])
                if sha256_file(source)!=original['sha256']:raise ValueError('Reference hash mismatch')
                references.append({'id':identifier,'style':name,'tier':tier,'role':'style_reference',
                    'path':f'references/{identifier}.jpg','sha256':original['sha256'],
                    'title':original['title'],'artist':original['artist_title'],
                    'is_public_domain':True,'selection_note':style['selection_note'],
                    'selection_authority':config['selection_authority'],'individual_human_rating':None,
                    'action_supervision_eligible':False,'source_record':original})
    cache=Path('artifacts/plasma_painter/free_sections')
    index=json.loads((cache/'index.json').read_text());targets=[];frames=[]
    for y in (0,18,31):
        record=next(r for r in index['records'] if r['y']==y)
        path=cache/f'section-y{y}.generated.json'
        if sha256_file(path)!=record['sha256']:raise ValueError('Plasma cache hash mismatch')
        clip=json.loads(path.read_text())
        if clip['split']!='art_train':raise ValueError('Only art_train targets')
        for n in (0,1):
            frame=next(f for f in clip['frames'] if f['source']['frame_index']==n)
            PaintingGym(frame)
            identifier=f'85604-y{y}-f{n}'
            targets.append({'id':identifier,'role':'scientific_target','shot':'85604','frame':n,
                'section':y,'split':'art_train','cache_sha256':record['sha256'],
                'frame_hash':stable_hash(frame),'image':f'targets/{identifier}.png',
                'features':f'targets/{identifier}.json','normalization':record['normalization'],
                'aesthetic_tier':None})
            frames.append(frame)
    out.mkdir(parents=True);(out/'references').mkdir();(out/'targets').mkdir()
    for ref in references:shutil.copy2(ref['source_record']['file'],out/ref['path'])
    for target,frame in zip(targets,frames):
        PaintingGym(frame).observe()['scientific'].save(out/target['image'])
        write_json(out/target['features'],frame)
        target['image_sha256']=sha256_file(out/target['image'])
        target['features_sha256']=sha256_file(out/target['features'])
    tasks=[{'id':f'{style}-{target["id"]}','style':style,'target':target['id'],
            'style_reference_ids':[r['id'] for r in references if r['style']==style and r['tier']=='love'],
            'generated_candidate_tier':'unrated','preference_label':None}
           for style in config['styles'] for target in targets]
    manifest={'version':'paired-style-targets-v1','git':git_state(),'roles':config['roles'],
        'selection_authority':config['selection_authority'],'individual_human_ratings':False,
        'library_sha256':sha256_file(library_path),'preprocessing':index['source_preprocessing_git'],
        'references':references,'targets':targets,'tasks':tasks,'reward_training_enabled':False}
    write_json(out/'manifest.json',manifest,overwrite=False)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    m=build(a.output);print(json.dumps({'style_love':sum(r['tier']=='love' for r in m['references']),
        'scientific_targets':len(m['targets']),'paired_tasks':len(m['tasks'])}))


if __name__=='__main__':main()
