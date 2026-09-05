"""Six hand-authored finite-mark studies; matched training data, not trained artists."""
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

STUDIES={'directional':'Directional bristle / Van Gogh-inspired',
         'pointillist':'Pointillism / Seurat-inspired',
         'cubist':'Angular facets / Picasso-inspired',
         'tonal':'Tonal brushwork / Manet-inspired',
         'pencil':'Pencil / cross-hatching',
         'screenprint':'Screenprint marks / Warhol-inspired'}


def main():
    started=time.perf_counter();config=load_config('configs/plasma_painter/pilot.yaml');root=artifact_root(config)
    index=json.loads((root/'features/index.json').read_text());entry=next(c for c in index['clips'] if c['split']=='art_train')
    clip=json.loads(Path(entry['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    frames=[with_stroke_samples(f) for f in clip['frames']]
    code=Path('plasma_painter/renderer/reference_renderers/finite_styles.js').read_text()
    output=root/'finite-style-atlas';output.mkdir(parents=True,exist_ok=True)
    boards=[Image.new('RGB',(1152,576),'#f7f0df') for _ in frames];records=[]
    for j,(key,label) in enumerate(STUDIES.items()):
        style={'study':key,'grain':0}
        trace=run_program(code,frames,style=style,seed=1701,profile='stroke_only')
        if not trace.valid: raise RuntimeError(f'{key}: {trace.error}')
        images=CanvasRuntime(384,256,style,1701).render_clip(frames,trace.operations_by_frame)
        x,y=(j%3)*384,(j//3)*288
        for i,img in enumerate(images):
            boards[i].paste(img,(x,y+32));ImageDraw.Draw(boards[i]).text((x+8,y+10),label,fill='#252525')
        images[0].save(output/f'{key}.png')
        imageio.mimsave(output/f'{key}.gif',[np.asarray(im) for im in images],duration=.35,loop=0)
        (output/f'{key}-operations.json').write_text(json.dumps(trace.operations_by_frame)+'\n')
        records.append({'study':key,'label':label,'marks_per_frame':[len(o)-1 for o in trace.operations_by_frame]})
        print(f'{key}: {len(images)} frames rendered',flush=True)
    boards[0].save(output/'comparison.png');imageio.mimsave(output/'comparison.gif',[np.asarray(b) for b in boards],duration=.35,loop=0)
    (output/'manifest.json').write_text(json.dumps({'status':'hand_authored_not_trained','profile':'stroke_only','field':'density_fluctuation','clip':clip['clip_id'],'indices':[f['source']['frame_index'] for f in frames],'program_hash':stable_hash(code),'records':records,'provenance':experiment_provenance(config,wall_seconds=time.perf_counter()-started,stage='finite_style_atlas')},indent=2)+'\n')
    print(output/'comparison.gif')


if __name__=='__main__':main()
