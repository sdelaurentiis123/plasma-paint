from pathlib import Path
from plasma_painter.config import stable_hash
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.renderer.sandbox import run_program
from scripts.render_style_atlas import STUDIES


def test_styles_have_distinct_geometry_and_are_reusable(synthetic_clip):
    frames=[with_stroke_samples(f) for f in synthetic_clip['frames']]
    code=Path('plasma_painter/renderer/reference_renderers/finite_styles.js').read_text()
    hashes=[]
    for key in STUDIES:
        a=run_program(code,frames,style={'study':key},profile='stroke_only',seed=17)
        b=run_program(code,frames,style={'study':key},profile='stroke_only',seed=17)
        assert a.valid,a.error
        assert a.operations_by_frame==b.operations_by_frame
        assert a.operations_by_frame[0]!=a.operations_by_frame[-1]
        hashes.append(stable_hash([op['args']['points'] for op in a.operations_by_frame[0] if op['op']=='mark']))
    assert len(set(hashes))==6
