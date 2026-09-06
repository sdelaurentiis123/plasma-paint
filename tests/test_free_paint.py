import numpy as np
import pytest
from plasma_painter.features.pipeline import encode_unit_raster
from plasma_painter.generation.free_paint_prompt import wrap_body
from plasma_painter.renderer.sandbox import run_program


def frame():
    return {'rasters':{'density':encode_unit_raster(np.broadcast_to(np.linspace(0,1,5)[:,None],(5,6)))}}


def test_free_strokes_are_not_anchored():
    code=wrap_body("api.createPaper({color:'#ffffff',grain:0}); api.paintStroke({points:[[.02,.1],[.5,.8],[.98,.3]],color:'#ff0033',width:.03,opacity:.6,medium:'ink'});")
    result=run_program(code,[frame()],profile='free_paint')
    assert result.valid,result.error
    assert 'sample_id' not in result.operations_by_frame[0][1]['args']
    assert not run_program(code,[frame()],profile='stroke_only').valid


def test_free_queries_and_one_sided_gradient():
    code=wrap_body("""api.createPaper({color:'#ffffff',grain:0});
      const v=api.sample('density',.5,.6), g=api.gradient('density',0,.5);
      if(Math.abs(v-.5)>.005 || Math.abs(g.dx-1)>.02 || Math.abs(g.dz)>.001) throw new Error('query mismatch');
      api.paintStroke({points:[[.1,.1],[.8,.8]],color:'#223344',opacity:v,width:.02,medium:'graphite'});""")
    result=run_program(code,[frame()],profile='free_paint')
    assert result.valid,result.error


def test_free_queries_do_not_exist_in_legacy_profile():
    code=wrap_body("api.sample('density',.5,.5); api.createPaper({color:'#ffffff',grain:0});")
    assert not run_program(code,[frame()]).valid


@pytest.mark.parametrize('target',['api.paintStroke','Math.sin','frameFeatures','styleConfig'])
def test_computed_constructor_cannot_reach_host_codegen(target):
    # Harmless arithmetic payload tests the realm boundary beyond regex filtering.
    code=wrap_body("const k='con'+'structor'; const f="+target+"[k][k]; f('return 7')(); api.createPaper({color:'#ffffff',grain:0}); api.paintStroke({points:[[.1,.1],[.3,.4]],color:'#123456',width:.02,opacity:.5});")
    result=run_program(code,[frame()],profile='free_paint')
    assert not result.valid
    assert 'Code generation from strings disallowed' in result.error,result.error


def test_api_errors_stay_in_vm_realm():
    code=wrap_body("try{api.sample('absent',.5,.5)}catch(e){const k='con'+'structor'; e[k][k]('return 7')();} api.createPaper({color:'#ffffff',grain:0}); api.paintStroke({points:[[.1,.1],[.3,.4]],color:'#123456',width:.02,opacity:.5});")
    result=run_program(code,[frame()],profile='free_paint')
    assert not result.valid
    assert 'Code generation from strings disallowed' in result.error,result.error


def test_free_paper_and_strokes_do_not_flicker_with_frame_number():
    from plasma_painter.renderer.canvas_runtime import CanvasRuntime
    code=wrap_body("api.createPaper({color:'#ffffff',grain:.01}); api.paintStroke({points:[[.1,.1],[.8,.8]],color:'#224466',medium:'charcoal',width:.03,opacity:.5,stroke_id:5});")
    frames=[{**frame(),'source':{'frame_index':i}} for i in (0,1)]
    r=run_program(code,frames,profile='free_paint');assert r.valid,r.error
    images=CanvasRuntime(128,96,{},17).render_clip(frames,r.operations_by_frame)
    assert images[0].tobytes()==images[1].tobytes()
