"""Package downloaded Rusty renders for local, synchronized visual review."""
import json
from pathlib import Path
from PIL import Image
from plasma_painter.config import stable_hash, write_json


def build_candidates():
    root = Path("artifacts/plasma_painter/rusty-schema-v2")
    static = Path(__file__).with_name("static")
    manifest = json.loads((root / "reference_pool/manifest.json").read_text())
    result = []
    for index, record in enumerate(sorted(manifest["records"], key=lambda r:r["seed"])):
        if record["shot"] != "85604" or record["split"] != "art_train":
            raise ValueError("candidate gallery requires permitted 85604 training clips")
        code = (root / "programs/candidates" / Path(record["program_path"]).name).read_text()
        if stable_hash(code) != record["program_hash"]:
            raise ValueError("rendered program hash does not match downloaded code")
        code_name = f"candidate-{index}.generated.txt"
        (static / code_name).write_text(code)
        frames = []
        for offset, original in enumerate(record["frames"]):
            relative = Path(original.split("/renders/", 1)[1])
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("invalid render path")
            image_path = root / "renders" / relative
            name = f"candidate-{index}-frame-{offset}.generated.webp"
            with Image.open(image_path) as image:
                image.convert("RGB").save(static / name, "WEBP", quality=90)
            frames.append(name)
        result.append({"label": chr(65+index), "seed":record["seed"],
                       "hash":record["program_hash"], "frames":frames,
                       "indices":record["frame_indices"], "code":code_name,
                       "clip_id":record["clip_id"]})
    write_json(static / "candidates.generated.json", {"job":"6985077", "candidates":result})
    print(f"Packaged {len(result)} verified model-generated painters")


if __name__ == "__main__":
    build_candidates()
