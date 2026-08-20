"""Inspect with base-only JSONL/in-memory trace sinks; prints redaction and event counts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from penampakan import InMemoryTraceSink, JsonlTraceSink, Penampakan
from penampakan.tracing import redact_trace_data


def main() -> None:
    memory = InMemoryTraceSink()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        with (
            Image.new("RGB", (24, 16), "gold") as image,
            Penampakan(
                trace_sinks=(memory, JsonlTraceSink(path)),
                owns_trace_sinks=True,
            ) as vision,
        ):
            vision.inspect(image)
        lines = path.read_text(encoding="utf-8").splitlines()
    safe = redact_trace_data(
        {"path": "/private/image.png", "question": "secret?", "api_key": "secret"}
    )
    print(f"events={len(memory.events)} jsonl_lines={len(lines)}")
    print(f"redacted={json.dumps(safe, sort_keys=True)}")


if __name__ == "__main__":
    main()
