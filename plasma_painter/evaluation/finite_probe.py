"""Training-only finite-mark diagnostics. Not the preregistered fidelity gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from plasma_painter.config import stable_hash, write_json
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.features.pipeline import decode_unit_raster
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.rewards.fidelity import pigment_density


def correlation(a,b):
    if np.std(a)<1e-8 or np.std(b)<1e-8: return 0.0
    value=float(spearmanr(np.ravel(a),np.ravel(b)).statistic)
    return value if np.isfinite(value) else 0.0


def change(images):
    arrays=[np.asarray(im,dtype=float)/255 for im in images]
    return float(np.mean([np.mean(np.abs(a-b)) for a,b in zip(arrays,arrays[1:])]))


def probe(code,clip,seed=1701,profile='stroke_only'):
    if clip['split']!='art_train' or not all(str(f['source']['shot'])=='85604' and
                0<=f['source']['frame_index']<288 for f in clip['frames']):
        raise ValueError('This diagnostic permits only old-85604 art_train frames within [0,288)')
    if profile not in {'stroke_only','free_paint'}:raise ValueError('Unsupported probe profile')
    frames=[with_stroke_samples(f) for f in clip['frames']] if profile=='stroke_only' else clip['frames']
    if len(frames)<2: raise ValueError('A temporal probe needs at least two frames')
    report={'program_hash':stable_hash(code),'clip_id':clip['clip_id'],'seed':seed,
            'frame_indices':[f['source']['frame_index'] for f in frames],
            'scope':'exploratory diagnostics; not a scientific-fidelity pass or aesthetic score'}
    result=run_program(code,frames,seed=seed,profile=profile)
    report.update(valid=result.valid,error=result.error)
    if not result.valid:return report
    repeat=run_program(code,frames,seed=seed,profile=profile)
    report['operations_reproducible']=repeat.valid and repeat.operations_by_frame==result.operations_by_frame
    style={'name':'finite-probe','grain':0,'paper':'#f7f0df'}
    images=CanvasRuntime(384,256,style,seed).render_clip(frames,result.operations_by_frame)
    repeated_images=CanvasRuntime(384,256,style,seed).render_clip(frames,result.operations_by_frame)
    report['pixels_reproducible']=all(a.tobytes()==b.tobytes() for a,b in zip(images,repeated_images))
    frozen=[{**frames[0],'source':f['source'],'time':f.get('time')} for f in frames]
    frozen_result=run_program(code,frozen,seed=seed,profile=profile)
    report['frozen_input_valid']=frozen_result.valid
    if frozen_result.valid:
        frozen_images=CanvasRuntime(384,256,style,seed).render_clip(frozen,frozen_result.operations_by_frame)
        report['frozen_input_mean_rgb_change']=change(frozen_images)
    report['evolving_input_mean_rgb_change']=change(images)
    scores=[]
    for frame,image,ops in zip(frames,images,result.operations_by_frame):
        field=abs(2*decode_unit_raster(frame['rasters']['density_fluctuation']).T-1)
        paper=next(op['args']['color'] for op in ops if op['op']=='createPaper')
        ink=pigment_density(image,field.shape,paper)
        # Sigma 2 cells, nonperiodic radial x / periodic display z.
        smooth=lambda a:gaussian_filter(a,2,mode=('wrap','nearest'))
        scores.append(correlation(smooth(ink),smooth(field)))
    report['coarse_abs_fluctuation_pigment_spearman']=float(np.mean(scores))
    report['per_frame_coarse_spearman']=scores
    report['operation_counts']=[len(ops) for ops in result.operations_by_frame]
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program',required=True)
    parser.add_argument('--clip',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',choices=['stroke_only','free_paint'],default='stroke_only')
    args=parser.parse_args()
    clip_path=Path(args.clip)
    # This command accepts an already-selected compact training clip, never raw shot files.
    permitted_section=clip_path.name in {f'section-y{y}.generated.json' for y in (0,1,16,17,18,19,30,31)}
    if not permitted_section and (not clip_path.name.startswith('tcv-85604-art_train-') or clip_path.suffix!='.json'):
        raise ValueError('Select a compact permitted training-clip JSON')
    clip=json.loads(clip_path.read_text())
    code=Path(args.program).read_text()
    records=[probe(code,clip,seed,args.profile) for seed in (1701,1702,1703)]
    write_json(args.output,{'records':records},overwrite=False)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
