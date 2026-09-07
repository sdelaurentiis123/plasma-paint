"""Build reference-conditioned reconstruction demonstrations using the gym prompt."""
import argparse,json,shutil
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from plasma_painter.config import write_json,sha256_file,git_state
from plasma_painter.generation.stroke_teacher import palette,target_image,fit
from plasma_painter.training.painting_gym import PaintingGym
from plasma_painter.training.gym_rollout import painting_prompt
from plasma_painter.rewards.fidelity import pigment_density,_safe_spearman
from plasma_painter.ratings.style_pool import resolve_style


def build(pool,output):
    path=Path(pool);pool_data=json.loads(path.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[];cases=[]
    for style in ('van-gogh','seurat-pointillism','manet'):
        references=resolve_style(path,style);reference_names=[]
        for ref in references:
            name=style+'-'+Path(ref).name;shutil.copy2(ref,out/name);reference_names.append(name)
        with Image.open(references[0]) as im:colors=palette(im)
        for target_record in pool_data['targets']:
            fpath=path.parent/target_record['features']
            assert sha256_file(fpath)==target_record['features_sha256']
            frame=json.loads(fpath.read_text());y=target_record['section'];n=target_record['frame']
            gym=PaintingGym(frame,size=(192,128),max_turns=32)
            target,field=target_image(frame,colors,(192,128))
            image,ops,losses=fit(target,field,budget=128)
            ink=pigment_density(image,field.shape,'#f7f0df')
            smooth=lambda a:gaussian_filter(a,2,mode=('wrap','nearest'))
            coarse=_safe_spearman(smooth(ink),smooth(abs(2*field-1)))
            # Reconstruction curriculum filter, not a full scientific validity claim.
            accepted=coarse>=.75 and losses[-1]<.3*losses[0]
            stem=f'{style}-y{y}-f{n}'
            image.save(out/f'{stem}-teacher.png');target.save(out/f'{stem}-target.png')
            write_json(out/f'{stem}-operations.json',ops)
            cases.append({'id':stem,'accepted':accepted,'coarse':coarse,'mse_ratio':losses[-1]/losses[0]})
            print('TEACHER',cases[-1],flush=True)
            if not accepted:continue
            previous=None
            for start in range(1,len(ops)-3,4):
                obs=gym.observe();strokes=[{k:v for k,v in op['args'].items() if k!='stroke_id'} for op in ops[start:start+4]]
                action={'action':'paint','strokes':strokes};images=list(reference_names)
                for kind in ('scientific','canvas'):
                    name=f'{stem}-t{gym.turn}-{kind}.png';obs[kind].save(out/name);images.append(name)
                prompt=painting_prompt(obs,previous)
                *_,previous=gym.step(action);assert previous['accepted'],previous
                rows.append({'id':f'{stem}-t{gym.turn}','style':style,'source':frame['source'],'section':y,
                    'split':'format_train' if n==0 else 'format_eval','images':images,'prompt':prompt,
                    'completion':json.dumps(action,separators=(',',':')),'origin':'greedy_reconstruction_teacher',
                    'aesthetic_training_eligible':False,'image_sha256':[sha256_file(out/f) for f in images]})
    (out/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(out/'manifest.json',{'git':git_state(),'pool_sha256':sha256_file(path),'cases':cases,
        'train_frame':0,'format_eval_frame':1,'purpose':'spatial reconstruction curriculum, NOT artist imitation',
        'count':len(rows),'reward_training_enabled':False,'full_fidelity_evaluated':False})
    if len({r['section'] for r in rows if r['split']=='format_train'})<3:
        raise ValueError('Need accepted reconstruction demonstrations in all three sections before training')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.pool,a.output)


if __name__=='__main__':main()
