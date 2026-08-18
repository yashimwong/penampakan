"""Reuse an OpenAI session and print cache/lineage counters.

Requires ``penampakan[openai]`` and ``OPENAI_API_KEY``.
"""

import asyncio
import os

from PIL import Image

from penampakan import (
    AgentSettings,
    AsyncPenampakan,
    CacheSettings,
    Capability,
    ColorsRequest,
    InspectionOperation,
    InspectionPlan,
    Settings,
)
from penampakan.llms import OpenAITextLLM


async def async_main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY after installing penampakan[openai].")
    settings = Settings(
        cache=CacheSettings(enabled=True),
        agent=AgentSettings(initial_capabilities=(Capability.METADATA, Capability.COLORS)),
    )
    plan = InspectionPlan(
        operations=(InspectionOperation(request=ColorsRequest()),),
        include_available_overview=False,
    )
    with Image.new("RGB", (96, 64), "seagreen") as image:
        async with OpenAITextLLM(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), api_key=api_key
        ) as llm:
            async with AsyncPenampakan(llm=llm, settings=settings) as vision:
                async with await vision.open_image(image) as session:
                    await session.inspect(plan)
                    cached = await session.inspect(plan)
                    answers = [
                        await session.ask(question)
                        for question in ("What color dominates?", "What are its dimensions?")
                    ]
                    lineage = {asset.id: asset.parent_id for asset in session.assets}
    print(f"answers={len(answers)} cache_hits={cached.trace.summary.cache_hits}")
    print(f"lineage={lineage}")
    print("evidence=" + ",".join(str(len(answer.evidence)) for answer in answers))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
