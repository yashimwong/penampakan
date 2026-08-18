"""Use local LiteLLM, pinned caption weights, OCR extras; prints answer and revisions.

Prerequisites: ``penampakan[litellm,transformers,ocr]``, Tesseract, an Ollama
server with llama3.1:8b, and the pinned BLIP snapshot in the Hugging Face cache.
"""

import asyncio

from PIL import Image, ImageDraw

from penampakan import AgentSettings, AsyncPenampakan, Capability, Settings
from penampakan.backends import TesseractBackend, TransformersCaptionBackend
from penampakan.llms import LiteLLMTextLLM

CAPTION_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"


async def async_main() -> None:
    caption = TransformersCaptionBackend(
        revision=CAPTION_REVISION,
        local_files_only=True,
    )
    ocr = TesseractBackend(languages=("eng",))
    settings = Settings(
        agent=AgentSettings(
            initial_capabilities=(Capability.METADATA, Capability.CAPTION, Capability.OCR)
        )
    )
    with Image.new("RGB", (320, 100), "white") as image:
        ImageDraw.Draw(image).text((20, 35), "LOCAL MODELS", fill="black")
        async with (
            LiteLLMTextLLM(
                model="ollama_chat/llama3.1:8b",
                allow_json_only=True,
                api_base="http://127.0.0.1:11434",
            ) as llm,
            AsyncPenampakan(
                llm=llm,
                backends=(caption, ocr),
                settings=settings,
            ) as vision,
        ):
            answer = await vision.ask(image, "What text and scene are visible?")
    print(f"status={answer.status.value}")
    print(f"caption_revision={caption.descriptor.model_revision}")
    print(f"answer={answer.answer}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
