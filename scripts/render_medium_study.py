"""Same permitted training clip, six bounded stroke tools; NOT trained outputs."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw

from plasma_painter.config import artifact_root, load_config
from plasma_painter.renderer.dsl import STROKE_MEDIA, validate_operations
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.canvas_runtime.runtime import RUNTIME_VERSION
from plasma_painter.renderer.sandbox import run_program


def main():
    config=load_config('configs/plasma_painter/pilot.yaml')
    index=json.loads((artifact_root(config)/'features/index.json').read_text())
    record=next(r for r in index['clips'] if r['split']=='art_train')
    clip=json.loads(Path(record['path']).read_text())
    assert clip['split']=='art_train'
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    style=yaml.safe_load(Path(config['renderer']['style_config']).read_text())
    code=Path(config['renderer']['baseline_program']).read_text()
    result=run_program(code,clip['frames'],style=style,seed=1701)
    assert result.valid,result.error
    output=artifact_root(config)/'medium-study';output.mkdir(exist_ok=True,parents=True)
    board=Image.new('RGB',(1152,560),'#f2ede2');draw=ImageDraw.Draw(board)
    for j,medium in enumerate(STROKE_MEDIA):
        traces=json.loads(json.dumps(result.operations_by_frame))
        for ops in traces:
            for op in ops:
                if op['op'] in ('strokePath','dryBrushPath'):
                    op['args'].update(medium=medium,pressure=.85,texture=.7,width=min(.08,op['args'].get('width',.003)*3),opacity=.7)
                elif op['op']=='washRegion': op['args']['opacity']=.045
                elif op['op'] in ('dab','poolPigment'): op['args']['opacity']=.08
            validate_operations(ops)
        images=CanvasRuntime(384,240,style,1701).render_clip(clip['frames'],traces)
        images[0].save(output/f'{medium}.png')
        imageio.mimsave(output/f'{medium}.gif',[np.asarray(im) for im in images],duration=.16,loop=0)
        x,y=(j%3)*384,(j//3)*280
        board.paste(images[0],(x,y+30));draw.text((x+12,y+8),medium.upper(),fill='#252525')
    board.save(output/'comparison.png')
    manifest={'status':'hand_authored_tool_study_not_trained','runtime':RUNTIME_VERSION,'clip':clip['clip_id'],'frame_indices':[f['source']['frame_index'] for f in clip['frames']],'shot':'85604','seed':1701,'media':STROKE_MEDIA,'feature_file':record['path'],'browser_pixel_equivalence':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output/'comparison.png')


if __name__=='__main__': main()
