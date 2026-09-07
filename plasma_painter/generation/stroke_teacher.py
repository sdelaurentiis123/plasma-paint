"""Greedy finite-stroke reconstruction teacher, not a learned artist or RL policy."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from plasma_painter.config import sha256_file,stable_hash,git_state,write_json
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.dsl import validate_operation,STROKE_MEDIA
from plasma_painter.features.pipeline import decode_unit_raster
from plasma_painter.rewards.fidelity import pigment_density,_safe_spearman


def palette(reference):
    """Two distinct actual quantized library colors; not an artist-specific palette."""
    q=reference.convert('RGB').resize((128,128)).quantize(colors=16)
    colors=np.asarray(q.getpalette()[:48],dtype=float).reshape(-1,3)
    chroma=colors.max(1)-colors.min(1)
    a=int(np.argmax(chroma))
    b=int(np.argmax(np.linalg.norm(colors-colors[a],axis=1)*(chroma+30)))
    return colors[[a,b]]


def target_image(frame,colors,size):
    field=decode_unit_raster(frame['rasters']['density_fluctuation']).T
    field=np.asarray(Image.fromarray(field.astype('float32')).resize(size,Image.Resampling.BILINEAR))
    paper=np.array([247,240,223.])
    amplitude=np.sqrt(abs(2*field-1)).clip(0,1)
    pigment=np.where((field>.5)[...,None],colors[0],colors[1])
    rgb=paper+(pigment-paper)*amplitude[...,None]
    return Image.fromarray(np.uint8(rgb.clip(0,255))),field


def fit(target,field,*,seed=1701,budget=128):
    if not 1<=budget<=256:raise ValueError('Teacher stroke budget 1..256')
    runtime=CanvasRuntime(*target.size,{},seed)
    paper={'op':'createPaper','args':{'color':'#f7f0df','grain':0}}
    paper_image=runtime.render_frame({},[paper]).convert('RGBA')
    current=Image.new('RGBA',target.size,(0,0,0,0));ops=[paper];losses=[]
    desired=np.asarray(target,dtype=float)/255
    canvas=np.asarray(paper_image.convert('RGB'),dtype=float)/255
    loss=float(np.mean((canvas-desired)**2));losses.append(loss)
    gy,gx=np.gradient(field);height,width=field.shape
    for step in range(budget):
        error=gaussian_filter(np.mean((canvas-desired)**2,axis=-1),2)
        row,col=np.unravel_index(np.argmax(error),error.shape)
        angle=np.arctan2(gx[row,col],-gy[row,col])
        center=np.array([col/(width-1),row/(height-1)])
        color='#'+''.join(f'{int(v):02x}' for v in np.asarray(target)[row,col])
        best=None;best_loss=loss
        for medium in STROKE_MEDIA:
            for length in (.08,.18):
                for brush_width in (.04,.08):
                    direction=np.array([np.cos(angle),np.sin(angle)])
                    points=[np.clip(center+t*length*direction,0,1).round(5).tolist() for t in (-.5,0,.5)]
                    args={'points':points,'medium':medium,'color':color,'width':brush_width,
                          'opacity':.8,'pressure':1.,'texture':.15,'stroke_id':step+1}
                    op={'op':'paintStroke','args':args};validate_operation(op,max_path_points=64)
                    layer=runtime._path_layer(args,dry=False,rng=np.random.default_rng(seed+(step+1)*7919))
                    combined=Image.alpha_composite(current,layer)
                    trial=np.asarray(Image.alpha_composite(paper_image,combined).convert('RGB'),dtype=float)/255
                    trial_loss=float(np.mean((trial-desired)**2))
                    if trial_loss<best_loss:
                        best=(op,combined,trial);best_loss=trial_loss
        if best is None:break
        op,current,canvas=best;ops.append(op);loss=best_loss;losses.append(loss)
    rendered=runtime.render_frame({},ops)
    assert np.array_equal(np.asarray(rendered),np.uint8(np.rint(canvas*255)))
    return rendered,ops,losses


def build(pool,output,budget=128):
    pool_path=Path(pool);manifest=json.loads(pool_path.read_text())
    out=Path(output);out.mkdir(parents=True,exist_ok=False);records=[]
    for style in ('van-gogh','seurat-pointillism','manet'):
        ref=next(r for r in manifest['references'] if r['style']==style and r['tier']=='love')
        image_path=pool_path.parent/ref['path'];assert sha256_file(image_path)==ref['sha256']
        with Image.open(image_path) as im:colors=palette(im)
        target_record=next(t for t in manifest['targets'] if t['section']==18 and t['frame']==0)
        feature_path=pool_path.parent/target_record['features']
        assert sha256_file(feature_path)==target_record['features_sha256']
        frame=json.loads(feature_path.read_text());assert frame['source']['shot']=='85604' and frame['source']['frame_index']==0
        target,field=target_image(frame,colors,(192,128))
        painting,ops,losses=fit(target,field,budget=budget)
        root=out/style;root.mkdir();target.save(root/'target.png');painting.save(root/'painting.png')
        ink=pigment_density(painting,field.shape,'#f7f0df')
        coarse=_safe_spearman(gaussian_filter(ink,2),gaussian_filter(abs(2*field-1),2))
        record={'style_reference':style,'reference_id':ref['id'],'reference_sha256':ref['sha256'],
            'source':frame['source'],'section':18,'frame_hash':stable_hash(frame),'palette':colors.tolist(),
            'mse_initial':losses[0],'mse_final':losses[-1],'losses':losses,
            'stroke_count':len(ops)-1,'media':dict(Counter(o['args']['medium'] for o in ops[1:])),
            'coarse_correspondence_diagnostic':coarse,'origin':'greedy_reconstruction_teacher',
            'learned_artist_style':False,'human_rating':None,'full_fidelity_evaluated':False,'seed':1701}
        write_json(root/'operations.json',ops);write_json(root/'manifest.json',record);records.append(record)
        print(style,record['stroke_count'],coarse,losses[0],losses[-1],flush=True)
    write_json(out/'manifest.json',{'git':git_state(),'records':records,'training_performed':False,
        'scope':'reference-derived colors and data-driven structure, NOT learned artist imitation'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',required=True);p.add_argument('--output',required=True)
    p.add_argument('--budget',type=int,default=128);a=p.parse_args();build(a.pool,a.output,a.budget)


if __name__=='__main__':main()
