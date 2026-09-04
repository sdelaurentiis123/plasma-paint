from pathlib import Path

from plasma_painter.generation.prompts import build_prompt
from plasma_painter.renderer.compiler import validate_program
from plasma_painter.renderer.sandbox import run_program


def test_schema_prompt_contains_executable_reference(config, synthetic_clip):
    prompt = build_prompt(synthetic_clip["frames"][0], schema_v2=True)
    code = Path(config["renderer"]["baseline_program"]).read_text()
    assert prompt.endswith(code)
    assert "ONE object argument" in prompt
    assert "vectors.density_gradient" in prompt
    assert "contour.path" in prompt  # explicit warning against observed hallucination
    result = run_program(code, synthetic_clip["frames"][:2], style={}, seed=17)
    assert result.valid, result.error


def test_equivalent_function_properties_work(config, synthetic_clip):
    code = Path(config["renderer"]["baseline_program"]).read_text()
    code = code.replace("reset(seed) {", "reset: function(seed) {")
    code = code.replace("renderFrame(frameFeatures, time, persistentState) {", "renderFrame: function(frameFeatures, time, persistentState) {")
    assert validate_program(code).valid
    assert run_program(code, synthetic_clip["frames"][:1]).valid
    assert not validate_program(code.replace("api.reset(seed)", "fetch('https://example.com')")).valid
