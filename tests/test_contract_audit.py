from pathlib import Path
import json
from plasma_painter.generation.contract_audit import CASES, make_prompt, validate_candidate
from plasma_painter.features.stroke_samples import with_stroke_samples


def test_ablation_changes_one_factor_at_a_time(synthetic_clip):
    reference = Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    plain = make_prompt(synthetic_clip['frames'][0], reference)
    example = make_prompt(synthetic_clip['frames'][0], reference, example=True)
    assert example.startswith(plain)
    assert example.endswith(reference)
    assert reference not in plain
    assert 'neutral_value' in plain and 'sample_count' in plain
    assert [(c[1],c[2],c[3]) for c in CASES] == [
        ('vision',False,False), ('vision',True,False), ('vision',True,True), ('coder',False,False)]


def test_infrastructure_timeout_retries_same_code(monkeypatch):
    calls=[]
    def fake(code,*args):
        calls.append(code)
        return {'error':'JavaScript syntax check timed out after 3 seconds'} if len(calls)==1 else {'accepted':True}
    monkeypatch.setattr('plasma_painter.generation.contract_audit.filter_candidate',fake)
    monkeypatch.setattr('plasma_painter.generation.contract_audit.subprocess.run',lambda *a,**kw:None)
    result=validate_candidate('same code',[],{}, {})
    assert calls == ['same code','same code']
    assert result['accepted'] and result['infrastructure_retry']['same_program']


def test_prompt_advertises_only_real_input_fields(synthetic_clip):
    frame=synthetic_clip['frames'][0]
    prompt=make_prompt(frame,'')
    payload=prompt.split('Runtime field subset (array truncated for illustration): ')[1].split('\n')[0]
    example=json.loads(payload)
    actual=with_stroke_samples(frame)
    assert set(example)<=set(actual)
    assert example['stroke_samples']==actual['stroke_samples'][:3]
    assert 'Begin literally: export function createPainter(api, styleConfig)' in prompt
    assert 'reset(seed) { api.reset(seed); }' in prompt
    assert 'NOT frameFeatures properties' in prompt
