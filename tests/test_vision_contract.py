from plasma_painter.generation.tool_contract import INTERFACE_GUIDANCE, repair_feedback
from plasma_painter.renderer.sandbox import run_program


def test_neutral_interface_example_is_executable():
    snippet=INTERFACE_GUIDANCE.split('one legal mark call is:\n')[1].split('\nEach call')[0]
    code='''export function createPainter(api, styleConfig) {
      return {reset(seed){api.reset(seed)},renderFrame(frameFeatures,time,persistentState){
        api.createPaper({color:'#f7f0df',grain:0});
        const s=frameFeatures.stroke_samples[0];
    '''+snippet+'}}}'
    result=run_program(code,[{'stroke_samples':[{'id':0,'x':.5,'z':.5}]}],profile='stroke_only')
    assert result.valid,result.error


def test_repair_feedback_preserves_error():
    assert 'test failure' in repair_feedback('test failure')
    assert 'Do not repeat it unchanged' in repair_feedback('test failure')
