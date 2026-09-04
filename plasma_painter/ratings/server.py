"""Launch a resumable local blind triage and pairwise rating interface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import uuid

from flask import Flask, abort, jsonify, request, send_file

from plasma_painter.config import artifact_root, load_config, stable_hash
from .schema import append_rating, read_ratings


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Plasma painter blind rating</title><style>
:root{color-scheme:light;background:#ede9df;color:#282924;font:16px/1.45 system-ui,sans-serif}body{margin:0}.shell{max-width:1280px;margin:auto;padding:20px}.top{display:flex;justify-content:space-between;align-items:baseline}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair.triage{grid-template-columns:minmax(300px,720px);justify-content:center}.pair.triage #right-view{display:none}.view{background:#f7f3e9;min-height:260px;display:grid;place-items:center}.view img{width:100%;height:auto;display:block}.actions{display:none;flex-wrap:wrap;gap:10px;justify-content:center;margin:20px 0}.actions.active{display:flex}button{font:inherit;padding:10px 16px;border:1px solid #555;background:#faf8f1;cursor:pointer}textarea{width:100%;min-height:52px;background:#faf8f1;border:1px solid #aaa}kbd{border:1px solid #aaa;padding:1px 5px}@media(max-width:700px){.pair{grid-template-columns:1fr}.view{min-height:0}}</style></head><body><main class="shell"><div class="top"><h1>Blind plasma-painter rating</h1><span id="progress"></span></div><p id="instructions">Triage the reference pool first. Identity is hidden.</p><div class="pair" id="pair"><div class="view"><img id="left" alt="Animated plasma painting"></div><div class="view" id="right-view"><img id="right" alt="Right animated plasma painting"></div></div><div class="actions" id="triage-actions"><button data-kind="triage" data-choice="love">Love</button><button data-kind="triage" data-choice="okay">Okay</button><button data-kind="triage" data-choice="reject">No</button></div><div class="actions" id="pair-actions"><button data-kind="pairwise" data-choice="left">Left</button><button data-kind="pairwise" data-choice="right">Right</button><button data-kind="pairwise" data-choice="tie">Tie</button><button data-kind="pairwise" data-choice="both_bad">Both bad</button></div><textarea id="comment" maxlength="500" placeholder="Optional short comment"></textarea><p id="status" role="status"></p></main><script>
let current=null;const pair=document.querySelector('#pair'),triage=document.querySelector('#triage-actions'),pairActions=document.querySelector('#pair-actions'),instructions=document.querySelector('#instructions');async function next(){const r=await fetch('/api/next');current=await r.json();if(current.done){document.querySelector('#status').textContent='No unseen items remain.';triage.classList.remove('active');pairActions.classList.remove('active');return}document.querySelector('#left').src=current.left.clip_url+'?n='+Date.now();document.querySelector('#progress').textContent=current.completed+' ratings saved';if(current.kind==='triage'){pair.classList.add('triage');triage.classList.add('active');pairActions.classList.remove('active');instructions.textContent='Triage this anonymous reference: love, okay, or no.'}else{pair.classList.remove('triage');triage.classList.remove('active');pairActions.classList.add('active');instructions.innerHTML='Same TCV clip and synchronized frames. Keys: <kbd>←</kbd> left, <kbd>→</kbd> right, <kbd>T</kbd> tie, <kbd>B</kbd> both bad.';document.querySelector('#right').src=current.right.clip_url+'?n='+Date.now()}document.querySelector('#status').textContent=current.repeated?'Consistency repeat':''}
async function rate(choice,kind){if(!current||current.done||kind!==current.kind)return;const comment=document.querySelector('#comment');const r=await fetch('/api/rate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:current.token,choice,comment:comment.value})});if(!r.ok){document.querySelector('#status').textContent=await r.text();return}comment.value='';await next()}document.querySelectorAll('button[data-choice]').forEach(b=>b.onclick=()=>rate(b.dataset.choice,b.dataset.kind));addEventListener('keydown',event=>{if(!current||current.kind!=='pairwise')return;const map={ArrowLeft:'left',ArrowRight:'right',t:'tie',T:'tie',b:'both_bad',B:'both_bad'};if(map[event.key])rate(map[event.key],'pairwise')});next();
</script></body></html>"""


def _family(record: dict) -> str:
    source = Path(record["program_path"]).read_text(encoding="utf-8")
    normalized = re.sub(r"#[0-9a-fA-F]{6}|\b\d+(?:\.\d+)?\b", "#", source)
    return stable_hash(normalized)


def _build_pairs(records: list[dict], config: dict) -> list[dict]:
    rng = random.Random(int(config["project"]["seed"]))
    by_clip: dict[str, list[dict]] = {}
    for record in records:
        by_clip.setdefault(record["clip_id"], []).append(record)
    pairs = []
    for clip_id, group in sorted(by_clip.items()):
        family_pair_counts: dict[tuple[str, str], int] = {}
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if left["program_hash"] == right["program_hash"]:
                    continue
                family_pair = tuple(sorted((_family(left), _family(right))))
                if family_pair_counts.get(family_pair, 0) >= 2:
                    continue
                family_pair_counts[family_pair] = family_pair_counts.get(family_pair, 0) + 1
                canonical = sorted([left["program_hash"], right["program_hash"]])
                pair_id = stable_hash({"clip": clip_id, "programs": canonical})
                ordered = [left, right]
                rng.shuffle(ordered)
                pairs.append({"kind": "pairwise", "pair_id": pair_id, "clip_id": clip_id, "left": ordered[0], "right": ordered[1], "repeated": False})
    rng.shuffle(pairs)
    repeat_count = max(1, round(len(pairs) * float(config["ratings"]["repeat_fraction"]))) if pairs else 0
    for source in pairs[:repeat_count]:
        copy = dict(source)
        copy["pair_id"] += "-repeat"
        copy["left"], copy["right"] = source["right"], source["left"]
        copy["repeated"] = True
        pairs.append(copy)
    return pairs


def create_app(config: dict) -> Flask:
    app = Flask(__name__)
    root = artifact_root(config)
    pool_path = root / "reference_pool" / "manifest.json"
    if not pool_path.exists():
        raise FileNotFoundError("reference pool missing; run plasma_painter.renderer.render_program first")
    records = json.loads(pool_path.read_text(encoding="utf-8"))["records"]
    rating_path = Path(config["ratings"]["jsonl_path"])
    if not rating_path.is_absolute():
        rating_path = Path.cwd() / rating_path
    active: dict[str, dict] = {}

    def public(record: dict) -> dict:
        return {"clip_url": "/artifact/" + str(Path(record["clip"]).resolve().relative_to(root)), "seed": record["seed"]}

    @app.get("/")
    def index():
        return PAGE

    @app.get("/artifact/<path:relative>")
    def artifact(relative: str):
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            abort(403)
        return send_file(target)

    @app.get("/api/next")
    def next_item():
        saved = read_ratings(rating_path)
        triage_records = [item for item in saved if item.get("kind") == "triage"]
        triaged = {item["reference_id"] for item in triage_records}
        candidate = next((item for item in records if item["reference_id"] not in triaged), None)
        if candidate:
            token = str(uuid.uuid4())
            active[token] = {"kind": "triage", "record": candidate}
            return jsonify({"done": False, "kind": "triage", "token": token, "completed": len(saved), "repeated": False, "left": public(candidate)})
        love = {item["reference_id"] for item in triage_records if item["choice"] == "love"}
        pairs = _build_pairs([item for item in records if item["reference_id"] in love], config)
        completed = {item.get("pair_id") for item in saved if item.get("kind") == "pairwise"}
        pair = next((item for item in pairs if item["pair_id"] not in completed), None)
        if pair is None:
            return jsonify({"done": True, "completed": len(saved), "love_references": len(love)})
        token = str(uuid.uuid4())
        active[token] = pair
        return jsonify({"done": False, "kind": "pairwise", "token": token, "completed": len(saved), "repeated": pair["repeated"], "left": public(pair["left"]), "right": public(pair["right"])})

    @app.post("/api/rate")
    def rate():
        payload = request.get_json(force=True)
        item = active.pop(payload.get("token"), None)
        if item is None:
            return "expired or unknown rating token", 409
        common = {"rating_id": str(uuid.uuid4()), "rater_id": config["ratings"]["rater_id"], "timestamp": datetime.now(timezone.utc).isoformat(), "comment": str(payload.get("comment", ""))[:500], "choice": payload.get("choice")}
        if item["kind"] == "triage":
            source = item["record"]
            record = {**common, "kind": "triage", "clip_id": source["clip_id"], "seed": source["seed"], "reference_id": source["reference_id"], "program_hash": source["program_hash"], "model_checkpoint": source["model_checkpoint"]}
        else:
            left, right = item["left"], item["right"]
            record = {**common, "kind": "pairwise", "pair_id": item["pair_id"], "clip_id": item["clip_id"], "seed": [left["seed"], right["seed"]], "left_program_hash": left["program_hash"], "right_program_hash": right["program_hash"], "left_checkpoint": left["model_checkpoint"], "right_checkpoint": right["model_checkpoint"], "order": [left["reference_id"], right["reference_id"]], "repeated": item["repeated"]}
        append_rating(rating_path, record)
        return jsonify({"saved": True})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    create_app(load_config(args.config)).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
