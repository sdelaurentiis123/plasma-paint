from pathlib import Path
from plasma_painter.generation.originality import check_construction


def test_copy_and_palette_edits_do_not_count(synthetic_clip):
    code=Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    frames=synthetic_clip['frames'][:2]
    assert not check_construction(code,code,frames)['different_construction']
    palette=code.replace('#19355c','#ff0000')
    assert not check_construction(palette,code,frames)['different_construction']
    changed=code.replace('pass*.18','pass*.38')
    assert check_construction(changed,code,frames)['different_construction']
