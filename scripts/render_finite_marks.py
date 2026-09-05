"""Real-data, untrained finite-mark brush/pencil construction study."""
import json
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from plasma_painter.config import load_config,artifact_root,stable_hash
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.provenance import experiment_provenance


def main():
    started=time.perf_counter()
    config=load_config('configs/plasma_painter/pilot.yaml');root=artifact_root(config)
    index=json.loads((root/'features/index.json').read_text())
    entry=next(c for c in index['clips'] if c['split']=='art_train')
    clip=json.loads(Path(entry['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    frames=[with_stroke_samples(f) for f in clip['frames']]
    code=Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    output=root/'finite-marks';output.mkdir(parents=True,exist_ok=True)
    boards=[Image.new('RGB',(768,288),'#f5efe1') for _ in frames];records=[]
    for column,medium in enumerate(('bristle','graphite')):
        style={'medium':medium,'grain':0}
        result=run_program(code,frames,style=style,seed=1701,profile='stroke_only')
        assert result.valid,result.error
        runtime=CanvasRuntime(384,256,style,1701)
        images=runtime.render_clip(frames,result.operations_by_frame)
        for i,img in enumerate(images):
            boards[i].paste(img,(384*column,32))
            ImageDraw.Draw(boards[i]).text((384*column+12,10),medium.upper()+' / finite marks only',fill='#252525')
        images[0].save(output/f'{medium}.png')
        imageio.mimsave(output/f'{medium}.gif',[np.asarray(im) for im in images],duration=.35,loop=0)
        # Show marks accumulating within one actual frame, separately from plasma time.
        ops=result.operations_by_frame[0]
        steps=[]
        for count in (50,150,300,600,len(ops)-1):
            im=CanvasRuntime(384,256,style,1701).render_frame(frames[0],ops[:count+1])
            ImageDraw.Draw(im).text((8,8),f'Stroke assembly: {min(count,len(ops)-1)} marks / fixed frame 0',fill='#252525');steps.append(np.asarray(im))
        imageio.mimsave(output/f'{medium}-assembly.gif',steps,duration=.65,loop=0)
        (output/f'{medium}-operations.json').write_text(json.dumps(result.operations_by_frame)+'\n')
        records.append({'medium':medium,'mark_counts':[len(o)-1 for o in result.operations_by_frame]})
    boards[0].save(output/'comparison.png')
    imageio.mimsave(output/'comparison.gif',[np.asarray(b) for b in boards],duration=.35,loop=0)
    (output/'manifest.json').write_text(json.dumps({'status':'hand_authored_not_trained','profile':'stroke_only','field':'density_fluctuation','program_hash':stable_hash(code),'clip':clip['clip_id'],'frame_indices':[f['source']['frame_index'] for f in frames],'seed':1701,'feature_file':entry['path'],'records':records,'provenance':experiment_provenance(config,wall_seconds=time.perf_counter()-started,stage='finite_mark_study')},indent=2)+'\n')
    print(output/'comparison.png')


if __name__=='__main__':main()
