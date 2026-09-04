"""Rebase only staged feature paths, retaining immutable source provenance."""
import json
from pathlib import Path

root = Path.cwd() / "artifacts/plasma_painter"
path = root / "features/index.json"
index = json.loads(path.read_text())
for clip in index["clips"]:
    clip["path"] = str(root / "features/clips" / Path(clip["path"]).name)
    if not Path(clip["path"]).is_file():
        raise FileNotFoundError(clip["path"])
index["normalization_path"] = str(root / "features/normalization.json")
path.write_text(json.dumps(index, indent=2) + "\n")
