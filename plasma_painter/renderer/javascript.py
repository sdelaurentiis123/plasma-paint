"""Parse untrusted JavaScript without executing it; fail closed on parser errors."""
import json
import subprocess
from pathlib import Path


def parse_source(code, mode='code'):
    if len(code.encode()) > 24000:
        raise ValueError('program exceeds 24000 bytes')
    try:
        result = subprocess.run(['node', str(Path(__file__).with_name('parse_program.mjs')), mode],
                                input=code, text=True, capture_output=True, timeout=3)
    except subprocess.TimeoutExpired as error:
        raise ValueError('JavaScript parser timed out after 3 seconds') from error
    if result.returncode:
        raise ValueError('JavaScript parse failed: '+result.stderr[-1500:])
    return json.loads(result.stdout)
