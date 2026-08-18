"""Inspect with base-only JSONL/in-memory trace sinks; prints redaction and event counts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from penampakan import Penampakan, TraceEvent
from penampakan.tracing import redact_trace_data


class MemorySink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        return None


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("w", encoding="utf-8")

    async def emit(self, event: TraceEvent) -> None:
        self.stream.write(event.model_dump_json() + "\n")
        self.stream.flush()

    async def aclose(self) -> None:
        self.stream.close()


def main() -> None:
    memory = MemorySink()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl"
        with (
            Image.new("RGB", (24, 16), "gold") as image,
            Penampakan(
                trace_sinks=(
                    memory,
                    JsonlSink(path),
                )
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
