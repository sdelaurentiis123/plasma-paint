"""Render matched base/adapter seeds on the same permitted cached training clip."""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw

from plasma_painter.config import artifact_root, load_config, stable_hash
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.sandbox import run_program


def main():
    config=load_config('configs/plasma_painter/pilot.yaml');root=artifact_root(config)
    index=json.loads((root/'features/index.json').read_text())
    entry=next(c for c in index['clips'] if c['split']=='art_train')
    clip=json.loads(Path(entry['path']).read_text())
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    style=yaml.safe_load(Path(config['renderer']['style_config']).read_text())
    boards=[Image.new('RGB',(768,1120),'#f2ede2') for _ in clip['frames']]
    records=[]
    for col,(folder,label,origin) in enumerate([
        ('rusty-media-sft','BASE','frozen_local_base_model'),
        ('rusty-adapter-preview','SFT ADAPTER','local_sft_adapter')]):
        manifest=json.loads((root/folder/'programs/sample_manifest-media_v3.json').read_text())
        for row,seed in enumerate(range(1701,1705)):
            record=next(r for r in manifest['records'] if r['seed']==seed)
            assert record['origin']==origin
            code=(root/folder/'programs/candidates'/Path(record['program_path']).name).read_text()
            assert stable_hash(code)==record['program_hash'] or (origin=='frozen_local_base_model' and stable_hash(code.rstrip())==record['program_hash'])
            result=run_program(code,clip['frames'],style=style,seed=seed)
            images=[]
            error=result.error
            if result.valid:
                try: images=CanvasRuntime(384,240,style,seed).render_clip(clip['frames'],result.operations_by_frame)
                except (ValueError,TypeError,KeyError,RuntimeError) as exc: error=str(exc)
            for i,board in enumerate(boards):
                x,y=col*384,row*280
                ImageDraw.Draw(board).text((x+12,y+10),f'{label} / seed {seed}',fill='#222222')
                if images: board.paste(images[i],(x,y+32))
                else: ImageDraw.Draw(board).text((x+12,y+90),'INVALID PROGRAM (not replaced)',fill='#aa2222')
            records.append({'method':label,'seed':seed,'program_hash':stable_hash(code),'rendered':bool(images),'error':error})
    output=root/'adapter-comparison';output.mkdir(parents=True,exist_ok=True)
    boards[0].save(output/'comparison.png')
    imageio.mimsave(output/'comparison.gif',[np.asarray(b) for b in boards],duration=.45,loop=0)
    (output/'manifest.json').write_text(json.dumps({'clip':clip['clip_id'],'status':'matched_training_clip_preview_not_held_out_evaluation','records':records},indent=2)+'\n')
    print(output/'comparison.gif')


if __name__=='__main__':main()
