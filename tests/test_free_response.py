import pytest
from plasma_painter.renderer.compiler import validate_program
from plasma_painter.generation.free_paint_prompt import prepare_response,wrap_body
from plasma_painter.renderer.sandbox import run_program


BODY="api.createPaper({color:'#ffffff',grain:0}); api.paintStroke({points:[[.1,.1],[.2,.2]],medium:'ink',color:'#123456',width:.01,opacity:.5});"


def test_comment_is_not_dynamic_code():
    assert validate_program(wrap_body('// Function eval fetch constructor\n'+BODY)).valid
    assert not validate_program(wrap_body('Function("return 7")();'+BODY)).valid


def test_template_expression_cannot_hide_code():
    assert not validate_program(wrap_body('const x=`${Function("return 7")()}`;'+BODY)).valid


def test_parser_removes_real_comments_only():
    assert not validate_program(wrap_body('const s="// comment"; Function("return 7")();'+BODY)).valid


@pytest.mark.parametrize('source',[BODY,'function renderFrame(frameFeatures,time,persistentState){'+BODY+'}',
    'function paintPlasmaDensity(frameFeatures,time,persistentState){'+BODY+'}'])
def test_body_and_callable_contract_execute(source):
    code,format=prepare_response(source)
    result=run_program(code,[{'rasters':{}}],profile='free_paint')
    assert result.valid,result.error
    assert len(result.operations_by_frame[0])==2


def test_wrong_function_signature_rejected_before_gpu_render():
    with pytest.raises(ValueError,match='exactly'):
        prepare_response('function painter(x){'+BODY+'}')
