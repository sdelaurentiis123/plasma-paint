from pathlib import Path
import pytest

from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.renderer.compiler import validate_program
from plasma_painter.renderer.sandbox import run_program


def test_reference_program_is_valid(config):
    code = Path(config["renderer"]["baseline_program"]).read_text()
    result = validate_program(code)
    assert result.valid, result.errors
    assert "strokePath" in result.api_calls


def test_forbidden_javascript_is_rejected():
    code = "export function createPainter(api, styleConfig){ fetch('https://bad'); return {reset(seed){api.reset(seed)},renderFrame(frameFeatures,time,persistentState){api.dab({center:[.5,.5],radius:.1,opacity:.2})}}}"
    result = validate_program(code)
    assert not result.valid
    assert any("network" in error for error in result.errors)


def test_constructor_escape_is_rejected():
    code = "export function createPainter(api, styleConfig){const x=Object['constructor'];return {reset(seed){api.reset(seed)},renderFrame(frameFeatures,time,persistentState){api.dab({center:[.5,.5],radius:.1,opacity:.2})}}}"
    result = validate_program(code)
    assert not result.valid
    assert any("prototype_escape" in error for error in result.errors)


def test_sandbox_times_out_bounded_child(synthetic_clip):
    code = "export function createPainter(api, styleConfig){return {reset(seed){api.reset(seed)},renderFrame(frameFeatures,time,persistentState){for(let i=0;i<1000000000;i+=1){} api.dab({center:[.5,.5],radius:.1,opacity:.2})}}}"
    result = run_program(code, synthetic_clip["frames"][:1], max_runtime_ms=80)
    assert not result.valid
    assert "timed out" in (result.error or "") or "timeout" in (result.error or "")


def test_rendering_is_deterministic(config, synthetic_clip):
    code = Path(config["renderer"]["baseline_program"]).read_text()
    result = run_program(code, synthetic_clip["frames"][:2], style={}, seed=11)
    assert result.valid, result.error
    style = {"paper": "#f2ede2", "grain": .03, "palette": {"low_density": "#233a4d", "high_density": "#d6aa62", "positive_fluctuation": "#9e4b3f", "negative_fluctuation": "#596b86", "mid_density": "#2f6f73"}}
    first = CanvasRuntime(160, 100, style, 11).render_clip(synthetic_clip["frames"][:2], result.operations_by_frame)
    second = CanvasRuntime(160, 100, style, 11).render_clip(synthetic_clip["frames"][:2], result.operations_by_frame)
    assert [image.tobytes() for image in first] == [image.tobytes() for image in second]


def candidate(factory='', reset='', render=''):
    return ('export function createPainter(api, styleConfig){'+factory+
            'return {reset(seed){'+reset+'},renderFrame(frameFeatures,time,persistentState){'+render+'}}}')


@pytest.mark.parametrize('phase', ['factory', 'reset'])
def test_drawing_outside_frame_is_not_silently_discarded(phase):
    code=candidate(**{phase:"api.createPaper({color:'#ffffff',grain:0});"})
    result=run_program(code,[{}])
    assert not result.valid
    assert 'drawing is only allowed inside renderFrame' in result.error


@pytest.mark.parametrize('phase', ['factory', 'reset'])
def test_initialization_is_timed(phase):
    code=candidate(**{phase:'for(let i=0;i<1000000000;i+=1){}'},
                   render="api.createPaper({color:'#ffffff',grain:0});")
    result=run_program(code,[{}],max_runtime_ms=80)
    assert not result.valid
    assert 'timed out' in result.error or 'timeout' in result.error


@pytest.mark.parametrize('expression, message', [
    ('[]', 'ONE options object'),
    ('{points:[{x:.4,z:.5},{x:.41,z:.5}],sample_id:0}', 'numeric [x,z] pairs'),
])
def test_mark_errors_are_actionable(expression,message):
    result=run_program(candidate(render='api.mark('+expression+');'),[{}])
    assert not result.valid
    assert message in result.error


def test_async_render_rejected():
    code=candidate(render='api.mark({});').replace('renderFrame(', 'async renderFrame(')
    assert 'forbidden JavaScript capability: asynchronous_code' in validate_program(code).errors
