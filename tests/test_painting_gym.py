import numpy as np
import pytest
from plasma_painter.training.painting_gym import PaintingGym
from plasma_painter.features.pipeline import encode_unit_raster


def env(**kw):
    f={'source':{'shot':'85604','frame_index':0},'rasters':{'density_fluctuation':encode_unit_raster(np.ones((8,8))*.5)}}
    return PaintingGym(f,size=(64,64),**kw)


ACTION={'action':'paint','strokes':[{'points':[[.1,.1],[.6,.8]],'medium':'ink','color':'#223344','width':.02,'opacity':.6}]}


def test_stroke_undo_and_repeat():
    e=env();blank=e.observe()['canvas'].tobytes()
    obs,*_=e.step(ACTION);paint=obs['canvas'].tobytes();assert paint!=blank
    assert e.step({'action':'undo'})[0]['canvas'].tobytes()==blank
    assert e.step(ACTION)[0]['canvas'].tobytes()==paint


def test_atomic_invalid_action_costs_turn():
    e=env(max_turns=1);a={'action':'paint','strokes':ACTION['strokes']+[{'points':[]}]} 
    _,reward,_,truncated,info=e.step(a)
    assert not info['accepted'] and reward==-1 and truncated and len(e.operations)==1
    with pytest.raises(RuntimeError):e.step(ACTION)


def test_no_code_and_blank_finish_fail():
    e=env();assert not e.step({'action':'eval','code':'anything'})[-1]['accepted']
    assert e.step({'action':'finish'})[1]==-1


def test_excluded_source_and_episode_limit():
    with pytest.raises(ValueError):PaintingGym({'source':{'shot':'85606','frame_index':0}})
    with pytest.raises(ValueError):env(max_strokes=100000)


def test_export_is_content_addressed_and_non_overwriting(tmp_path):
    import json
    from plasma_painter.config import stable_hash
    e=env();e.step(ACTION);e.step({'action':'finish'});out=e.save(tmp_path/'episode')
    assert json.loads((out/'manifest.json').read_text())['program_hash']==stable_hash(json.loads((out/'program.json').read_text()))
    with pytest.raises(FileExistsError):e.save(out)
