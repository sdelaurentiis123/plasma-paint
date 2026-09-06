import pytest
from plasma_painter.renderer.paint_program import compile_paint_program,validate_paint_program
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.features.stroke_samples import with_stroke_samples


def fixture():
    # Synthetic interface fixture, not an artist preset or generation context.
    return {'version':1,'paper':{'color':'#ffffff','grain':0},'layers':[
        {'tool':'ink','palette':['#102030','#987654'],'color_field':'value',
         'path':[[-1,0],[0,.2],[1,0]],'direction':'tangent','stride':1,'phase':0,'select':[0,1],
         'length':{'base':.01,'strength':.02},'width':.003,'opacity':.6,
         'pressure':.6,'texture':.3,'angle_offset':0}]}


def test_program_compiles_to_valid_data_responsive_marks(synthetic_clip):
    program=fixture();code=compile_paint_program(program)
    result=run_program(code,[with_stroke_samples(f) for f in synthetic_clip['frames']],profile='stroke_only')
    assert result.valid,result.error
    assert result.operations_by_frame[0]!=result.operations_by_frame[-1]
    assert all(op['args']['medium']=='ink' for frame in result.operations_by_frame for op in frame if op['op']=='mark')


def test_model_chooses_medium_and_palette():
    p=fixture();a=compile_paint_program(p)
    p['layers'][0]['tool']='charcoal';p['layers'][0]['palette']=['#000000','#ffffff']
    b=compile_paint_program(p)
    assert a!=b and 'charcoal' in b and 'ink' not in b


def test_rejects_injection_and_excess_budget():
    p=fixture();p['layers'][0]['tool']='fetch'
    with pytest.raises(ValueError):validate_paint_program(p)
    p=fixture();p['layers']*=3
    with pytest.raises(ValueError):validate_paint_program(p)
    p=fixture();p['layers'][0]['width']={'eval':'bad'}
    with pytest.raises(ValueError):validate_paint_program(p)
