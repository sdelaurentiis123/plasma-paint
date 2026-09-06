from pathlib import Path
from plasma_painter.generation.contract_audit import CASES, make_prompt, validate_candidate


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
