from __future__ import annotations

from pathlib import Path

import tomllib
from huggingface_hub import snapshot_download

manifest = tomllib.loads(Path(__file__).with_name("environment.toml").read_text(encoding="utf-8"))
for model in manifest["models"].values():
    snapshot_download(model["id"], revision=model["revision"])
