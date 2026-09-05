"""Stage six already-downloaded museum-cleared references for local vision context."""
import json
import shutil
from pathlib import Path
from plasma_painter.config import sha256_file

root=Path('artifacts/plasma_painter/vision-reference-stage');root.mkdir(parents=True,exist_ok=True)
pool=json.loads(Path('data/art_references/aic/manifest.json').read_text())['records']
ids=[28560,14586,27992,20199,44892,81533]
records=[]
for identifier in ids:
    r=next(r for r in pool if r['id']==identifier)
    assert r['is_public_domain'] is True and sha256_file(Path(r['file']))==r['sha256']
    shutil.copy2(r['file'],root/Path(r['file']).name);records.append(r)
(root/'manifest.json').write_text(json.dumps({'purpose':'inference reference context only','records':records},indent=2)+'\n')
print(root)
