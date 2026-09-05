from pathlib import Path
import pytest
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.renderer.sandbox import run_program
from plasma_painter.renderer.dsl import validate_operation,validate_stroke_only


def mark():
    return {'op':'mark','args':{'points':[[.48,.5],[.52,.5]],'width':.01,'color':'#304060','sample_id':0}}


def test_mark_limits():
    validate_operation(mark())
    for key,value in [('points',[[0,0],[1,1]]),('width',.08),('color','bad')]:
        op=mark();op['args'][key]=value
        with pytest.raises(ValueError):validate_operation(op)


def test_no_wash_or_fake_anchor():
    frame={'stroke_samples':[{'id':0,'x':.5,'z':.5}]}
    paper={'op':'createPaper','args':{'color':'#ffffff','grain':0}}
    validate_stroke_only([paper,mark()],frame)
    with pytest.raises(ValueError):validate_stroke_only([paper,{'op':'washRegion'}],frame)
    frame['stroke_samples'][0]['x']=.1
    with pytest.raises(ValueError):validate_stroke_only([paper,mark()],frame)


def test_reusable_mark_program(synthetic_clip):
    frames=[with_stroke_samples(f) for f in synthetic_clip['frames']]
    code=Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    for medium in ('bristle','graphite'):
        result=run_program(code,frames,style={'medium':medium},profile='stroke_only')
        assert result.valid,result.error
        assert result.operations_by_frame[0]!=result.operations_by_frame[-1]
        assert all(op['op'] in ('createPaper','mark') for ops in result.operations_by_frame for op in ops)
