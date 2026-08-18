"""Ask with ``penampakan[openai]`` and OPENAI_API_KEY; prints answer and evidence IDs."""

import asyncio
import os

from PIL import Image

from penampakan import AgentSettings, AsyncPenampakan, Capability, Settings
from penampakan.llms import OpenAITextLLM


async def async_main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY after installing penampakan[openai].")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    settings = Settings(
        agent=AgentSettings(initial_capabilities=(Capability.METADATA, Capability.COLORS))
    )
    with Image.new("RGB", (80, 50), "royalblue") as image:
        async with OpenAITextLLM(model=model, api_key=api_key) as llm:
            async with AsyncPenampakan(llm=llm, settings=settings) as vision:
                answer = await vision.ask(image, "What is the dominant color?")
    print(f"status={answer.status.value}")
    print(f"answer={answer.answer}")
    print("evidence=" + ",".join(item.observation.id for item in answer.evidence))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
