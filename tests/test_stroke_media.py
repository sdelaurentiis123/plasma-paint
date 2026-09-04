import hashlib

import pytest

from plasma_painter.renderer.dsl import STROKE_MEDIA, validate_operation
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.generation.prompts import build_prompt


def operation(medium):
    return {"op": "strokePath", "args": {"points": [[.1,.2],[.4,.7],[.9,.3]], "width": .07, "opacity": .8, "medium": medium, "texture": .8}}


def test_media_distinct_deterministic(synthetic_clip):
    hashes=[]
    for medium in STROKE_MEDIA:
        op=validate_operation(operation(medium))
        a=CanvasRuntime(192,128,{"grain":0},17).render_frame(synthetic_clip["frames"][0],[op])
        b=CanvasRuntime(192,128,{"grain":0},17).render_frame(synthetic_clip["frames"][0],[op])
        assert a.tobytes()==b.tobytes()
        hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
    assert len(set(hashes))==len(STROKE_MEDIA)


@pytest.mark.parametrize("key,value", [("medium","eval"),("pressure",2),("texture",-1)])
def test_medium_bounds(key,value):
    op=operation("ink");op["args"][key]=value
    with pytest.raises(ValueError): validate_operation(op)


def test_media_prompt(synthetic_clip):
    prompt=build_prompt(synthetic_clip["frames"][0],media=True)
    assert "MEDIUM STUDY" in prompt and "Make modest" not in prompt
    assert "EXACT RUNTIME CONTRACT" in prompt
