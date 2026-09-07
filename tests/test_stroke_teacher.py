import numpy as np
from PIL import Image
from plasma_painter.generation.stroke_teacher import fit


def test_teacher_improves_actual_raster_and_replays():
    field=np.ones((32,48),dtype=float)*.6
    target=Image.new('RGB',(48,32),'#859dad')
    a,ops,loss=fit(target,field,budget=8)
    b,ops2,loss2=fit(target,field,budget=8)
    assert ops==ops2 and a.tobytes()==b.tobytes()
    assert len(ops)>1 and loss[-1]<loss[0]
    assert all(y<x for x,y in zip(loss,loss[1:]))
