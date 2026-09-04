"""Build documented field-aligned sections from permitted training data."""
import argparse
from pathlib import Path
from plasma_painter.config import load_config, write_json, artifact_root, git_state
from plasma_painter.data.loaders import load_fixed_plane
from plasma_painter.features.pipeline import FeaturePipeline


def build_sections(config):
    static = Path(__file__).with_name("static")
    records = []
    for y in (0, 1, 16, 17, 18, 19, 30, 31):
        dc = config["data"]
        start, stop = dc["art_split"]["art_train"]
        keys = {key: value.rsplit("_y", 1)[0] + f"_y{y}" for key, value in dc["field_keys"].items()}
        data = load_fixed_plane(dc["source_path"], dc["geometry_path"],
                               shot="85604", field_keys=keys, stop=stop, plane_y=y)
        pipeline = FeaturePipeline(config["features"])
        normalization = pipeline.fit(data, start, stop)
        clip = pipeline.transform_clip(data, start, start+8, f"tcv-85604-section-y{y}-0000-0007", "art_train")
        clip["seed"] = config["project"]["seed"]
        for frame in clip["frames"]:
            frame["geometry"]["valid_domain"] = f"interior fixed field-aligned section y={y}; x guards removed"
        filename = f"section-y{y}.generated.json"
        write_json(static / filename, clip)
        records.append({"y": y, "url": filename, "label": f"Field-aligned section y={y}",
                        "geometry": clip["frames"][0]["geometry"], "normalization": normalization})
        print(f"Built section y={y}", flush=True)
    manifest = {"shot": "85604", "source_path": config["data"]["source_path"],
                "source_sha256": config["data"]["source_sha256"], "geometry_path": config["data"]["geometry_path"],
                "geometry_sha256": config["data"]["geometry_sha256"], "frames": [0,8],
                "normalization_frames": config["data"]["art_split"]["art_train"],
                "coordinate_note": "x radial; z periodic field-aligned, not a Cartesian R-Z slice; each section normalized separately",
                "git": git_state(), "sections": records}
    write_json(static / "sections.generated.json", manifest)
    write_json(artifact_root(config) / "web_demo/sections_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    build_sections(load_config(parser.parse_args().config))
