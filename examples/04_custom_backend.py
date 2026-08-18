"""Run base-only deterministic fakes; prints an answered result and evidence ID."""

import json

from PIL import Image

from penampakan import (
    AgentSettings,
    BackendDescriptor,
    Capability,
    CapabilityDescriptor,
    CaptionPayload,
    ObservationDraft,
    Penampakan,
    Settings,
    VisionResult,
)
from penampakan.backends import CallableVisionBackend
from penampakan.llms import CallableTextLLM


def main() -> None:
    descriptor = BackendDescriptor(
        name="example.caption",
        version="1.0",
        capabilities=(CapabilityDescriptor(capability=Capability.CAPTION),),
    )
    backend = CallableVisionBackend(
        descriptor,
        lambda image, request: VisionResult(
            observations=(ObservationDraft(payload=CaptionPayload(text="A red square.")),)
        ),
    )
    action = {
        "type": "answer",
        "status": "answered",
        "answer": "The image is a red square.",
        "evidence": [{"observation_id": "obs_000001", "supports": "Caption"}],
    }
    llm = CallableTextLLM(lambda request: json.dumps(action))
    settings = Settings(agent=AgentSettings(initial_capabilities=(Capability.CAPTION,)))
    with (
        Image.new("RGB", (32, 32), "red") as image,
        Penampakan(
            llm=llm,
            backends=(backend,),
            settings=settings,
            owns_llm=True,
        ) as vision,
    ):
        answer = vision.ask(image, "What is shown?")
    print(f"status={answer.status.value}")
    print(f"answer={answer.answer}")
    print(f"evidence={answer.evidence[0].observation.id}")


if __name__ == "__main__":
    main()
