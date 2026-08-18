"""Produce a deterministic base-only insufficient-evidence answer and budget summary."""

import json

from PIL import Image

from penampakan import AgentSettings, Capability, Penampakan, RunLimits, Settings
from penampakan.llms import CallableTextLLM


def main() -> None:
    action = {
        "type": "answer",
        "status": "insufficient_evidence",
        "answer": "The available observations cannot establish a serial number.",
        "evidence": [],
        "uncertainties": ["No text observation is available."],
    }
    llm = CallableTextLLM(lambda request: json.dumps(action))
    limits = RunLimits(
        max_steps=1,
        max_llm_calls=2,
        max_tool_calls=1,
        max_backend_calls=1,
        max_parallel_tools=1,
    )
    settings = Settings(
        agent=AgentSettings(initial_capabilities=(Capability.METADATA,)), run=limits
    )
    with (
        Image.new("RGB", (20, 20), "white") as image,
        Penampakan(llm=llm, settings=settings, owns_llm=True) as vision,
    ):
        answer = vision.ask(image, "What is the device serial number?")
    print(f"status={answer.status.value}")
    print(f"stop_reason={answer.trace.summary.stop_reason}")
    print(
        f"evidence={len(answer.evidence)} "
        f"llm_calls={answer.trace.summary.llm_calls}/{limits.max_llm_calls}"
    )


if __name__ == "__main__":
    main()
