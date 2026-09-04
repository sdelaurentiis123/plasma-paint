import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.compare_adapter_preview import remove_trailing_wrapper
from plasma_painter.renderer.compiler import validate_program


def test_wrapper_removal_is_narrow():
    code='export function createPainter(api, styleConfig) {\n}\n'
    assert remove_trailing_wrapper(code+'```\n')==code
    assert remove_trailing_wrapper(code+'This script explains the result.')==code
    assert remove_trailing_wrapper(code+'otherJavaScript();')==code+'otherJavaScript();'


def test_syntax_timeout_is_recorded(config):
    code=Path(config['renderer']['baseline_program']).read_text()
    with patch('plasma_painter.renderer.compiler.subprocess.run',side_effect=subprocess.TimeoutExpired('node',3)):
        result=validate_program(code)
    assert not result.valid
    assert any('timed out' in error for error in result.errors)
