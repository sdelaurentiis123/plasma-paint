"""Build and serve the local homepage demo without deploying it."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, send_from_directory

from plasma_painter.config import load_config
from .build_demo import build


def create_app(config: dict) -> Flask:
    build(config)
    static = Path(__file__).with_name("static")
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/<path:relative>")
    def files(relative: str):
        return send_from_directory(static, relative)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8088); args = parser.parse_args()
    create_app(load_config(args.config)).run(host=args.host, port=args.port, debug=False); return 0


if __name__ == "__main__":
    raise SystemExit(main())
