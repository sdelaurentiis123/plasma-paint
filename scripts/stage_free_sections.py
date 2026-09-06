"""Stage only existing permitted compact sections and their control measurements."""
import json
import shutil
from pathlib import Path
from plasma_painter.config import sha256_file,write_json
from plasma_painter.evaluation.finite_probe import probe


def main():
    out=Path('artifacts/plasma_painter/free_sections');out.mkdir(parents=True,exist_ok=False)
    reference=Path('plasma_painter/renderer/reference_renderers/finite_marks.js').read_text()
    source_manifest=json.loads(Path('artifacts/plasma_painter/web_demo/sections_manifest.json').read_text())
    records=[]
    for y in (0,18,31):
        path=Path(f'plasma_painter/web/demo/static/section-y{y}.generated.json')
        clip=json.loads(path.read_text())
        assert clip['split']=='art_train' and all(str(f['source']['shot'])=='85604' and 0<=f['source']['frame_index']<288 for f in clip['frames'])
        control=probe(reference,clip,1701)
        assert control['valid'] and control['operations_reproducible']
        shutil.copy2(path,out/path.name)
        meta=next(r for r in source_manifest['sections'] if r['y']==y)
        records.append({'y':y,'path':path.name,'sha256':sha256_file(path),'normalization':meta['normalization'],
                        'control':control,'coarse_floor':.95*control['coarse_abs_fluctuation_pigment_spearman']})
        write_json(out/'index.json',{'records':records,'source_manifest_sha256':sha256_file('artifacts/plasma_painter/web_demo/sections_manifest.json'),
                                   'source_preprocessing_git':source_manifest['git'],'status':'complete' if len(records)==3 else 'in_progress'})
        print(f'Staged permitted section y={y}, control correlation={control["coarse_abs_fluctuation_pigment_spearman"]:.3f}',flush=True)


if __name__=='__main__':main()
