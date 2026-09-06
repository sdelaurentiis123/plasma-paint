"""CPU-only interface test; the control is NOT an artist exemplar or model output."""
import json
from pathlib import Path
from plasma_painter.config import sha256_file
from plasma_painter.generation.free_paint_prompt import prepare_response
from plasma_painter.renderer.sandbox import run_program


code,_=prepare_response('''function renderFrame(frameFeatures,time,persistentState){
// Function comment must not be confused with executable dynamic code.
api.createPaper({color:'#ffffff',grain:0});
const v=api.sample('density_fluctuation',.317,.613);
api.paintStroke({points:[[.05,.1],[.5,v],[.95,.8]],medium:'ink',
color:'#123456',width:.01,opacity:.5});
}''')
root=Path('artifacts/plasma_painter/free_sections')
index=json.loads((root/'index.json').read_text())
for record in index['records']:
    path=root/record['path']
    assert path.parent==root and sha256_file(path)==record['sha256']
    clip=json.loads(path.read_text())
    assert clip['split']=='art_train'
    assert all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
    result=run_program(code,clip['frames'],profile='free_paint')
    assert result.valid,result.error
    print('CPU interface preflight passed:',record['y'],len(clip['frames']),flush=True)
