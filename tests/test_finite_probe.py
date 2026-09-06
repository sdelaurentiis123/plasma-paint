import pytest
from plasma_painter.evaluation.finite_probe import correlation, probe


def test_known_correlations():
    assert correlation([0,1,2],[0,1,2])==1
    assert correlation([0,1,2],[2,1,0])==-1
    assert correlation([1,1,1],[0,1,2])==0


def test_probe_rejects_non_training_clip_before_execution():
    with pytest.raises(ValueError):probe('',{'split':'art_test','frames':[]})


def test_finite_marks_cannot_silently_use_legacy_rl_rewards():
    from plasma_painter.training.environment import evaluate_program
    with pytest.raises(ValueError,match='Legacy RL fidelity metrics'):
        evaluate_program('',{}, {'renderer':{'profile':'stroke_only'}},seed=0)
