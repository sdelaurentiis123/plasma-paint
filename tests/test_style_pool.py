import json
import pytest
from plasma_painter.config import sha256_file
from plasma_painter.ratings.style_pool import resolve_style


def test_only_style_love_is_selected_and_verified(tmp_path):
    f=tmp_path/'ref.jpg';f.write_bytes(b'test-fixture-not-image')
    ref={'style':'s','tier':'love','path':f.name,'role':'style_reference','is_public_domain':True,'sha256':sha256_file(f)}
    p=tmp_path/'manifest.json'
    p.write_text(json.dumps({'version':'paired-style-targets-v1','references':[ref,{**ref,'tier':'okay'}]}))
    assert resolve_style(p,'s')==[str(f)]
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):resolve_style(p,'s')


def test_scientific_target_cannot_become_style_love(tmp_path):
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'version':'paired-style-targets-v1','references':[
        {'style':'s','tier':'love','path':'x.png','role':'scientific_target','is_public_domain':True}]}))
    with pytest.raises(ValueError):resolve_style(p,'s')
