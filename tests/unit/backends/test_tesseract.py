from __future__ import annotations

import asyncio
import importlib
import threading
from collections.abc import Mapping, Sequence
from io import BytesIO
from types import SimpleNamespace
from typing import cast

import pytest
from PIL import Image

from penampakan.backends.tesseract import TesseractBackend
from penampakan.errors import BackendUnavailableError, SessionClosedError
from penampakan.models import (
    BackendImage,
    Box,
    Capability,
    CaptionRequest,
    ImageAsset,
    OCRRequest,
    TextPayload,
)


def _backend_image(width: int = 100, height: int = 100) -> BackendImage:
    source = Image.new("RGB", (width, height), "white")
    output = BytesIO()
    try:
        source.save(output, format="PNG")
        content = output.getvalue()
    finally:
        output.close()
        source.close()
    return BackendImage(
        asset=ImageAsset(
            id="img_aaaaaaaaaaaaaaaa",
            width=width,
            height=height,
            mode="RGB",
            mime_type="image/png",
            original_format="PNG",
            digest_sha256="a" * 64,
            parent_id=None,
            derivation_depth=0,
            transform=None,
        ),
        content=content,
    )


def _data(**overrides: Sequence[object]) -> dict[str, Sequence[object]]:
    values: dict[str, Sequence[object]] = {
        "page_num": [1],
        "block_num": [1],
        "par_num": [1],
        "line_num": [1],
        "word_num": [1],
        "left": [10],
        "top": [20],
        "width": [30],
        "height": [10],
        "conf": [90],
        "text": ["hello"],
    }
    values.update(overrides)
    return values


class FakePytesseract:
    def __init__(
        self,
        *,
        data: Mapping[str, Sequence[object]] | None = None,
        languages: Sequence[str] = ("eng",),
    ) -> None:
        self.Output = SimpleNamespace(DICT=object())
        self.pytesseract = SimpleNamespace(tesseract_cmd="system-tesseract")
        self.data = dict(data or _data())
        self.languages = list(languages)
        self.version_calls = 0
        self.language_calls = 0
        self.analysis_calls = 0
        self.analysis_arguments: list[tuple[tuple[int, int], str, str, object, str]] = []

    def get_tesseract_version(self) -> str:
        self.version_calls += 1
        return "5.4.1"

    def get_languages(self, config: str = "") -> list[str]:
        self.language_calls += 1
        return list(self.languages)

    def image_to_data(
        self,
        image: Image.Image,
        *,
        lang: str,
        config: str,
        output_type: object,
    ) -> Mapping[str, Sequence[object]]:
        self.analysis_calls += 1
        self.analysis_arguments.append(
            (image.size, lang, config, output_type, self.pytesseract.tesseract_cmd)
        )
        return self.data


def _install(monkeypatch: pytest.MonkeyPatch, fake: object) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)


def test_construction_and_capability_discovery_do_not_import_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    backend = TesseractBackend(
        executable="/configured/tesseract",
        languages=("eng", "deu"),
        config="--oem 1",
        max_concurrency=3,
    )
    descriptor = backend.descriptor

    assert imports == []
    assert descriptor.name == "tesseract"
    assert descriptor.model_id is None
    assert descriptor.durable_cache_eligible is True
    assert descriptor.version.startswith("1.0+lang.eng+deu+config.")
    assert descriptor.max_concurrency == 3
    assert descriptor.capabilities[0].capability is Capability.OCR
    assert descriptor.capabilities[0].features == frozenset({"ocr.languages", "ocr.word_boxes"})
    assert backend.supports(OCRRequest())
    assert backend.supports(OCRRequest(languages=("deu",)))
    assert not backend.supports(OCRRequest(languages=("fra",)))
    assert not backend.supports(CaptionRequest())


@pytest.mark.asyncio
async def test_missing_extra_diagnostic_is_actionable_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def missing(name: str) -> object:
        nonlocal calls
        calls += 1
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    backend = TesseractBackend()

    for _ in range(2):
        with pytest.raises(BackendUnavailableError) as raised:
            await backend.analyze(_backend_image(), OCRRequest())
        assert raised.value.code == "ocr_extra_missing"
        assert raised.value.backend_name == "tesseract"

    assert calls == 1


@pytest.mark.asyncio
async def test_missing_binary_and_language_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = FakePytesseract()

    def missing_binary() -> str:
        raise FileNotFoundError

    monkeypatch.setattr(binary, "get_tesseract_version", missing_binary)
    _install(monkeypatch, binary)
    backend = TesseractBackend()

    with pytest.raises(BackendUnavailableError) as binary_error:
        await backend.analyze(_backend_image(), OCRRequest())

    assert binary_error.value.code == "tesseract_binary_unavailable"

    languages = FakePytesseract(languages=("eng",))
    _install(monkeypatch, languages)
    missing_language = TesseractBackend(languages=("deu",))

    with pytest.raises(BackendUnavailableError) as language_error:
        await missing_language.analyze(_backend_image(), OCRRequest())

    assert language_error.value.code == "tesseract_language_unavailable"


@pytest.mark.asyncio
async def test_words_are_grouped_sorted_and_weighted_by_character_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract(
        data=_data(
            page_num=[1, 1, 1],
            block_num=[1, 1, 1],
            par_num=[1, 1, 1],
            line_num=[2, 1, 1],
            word_num=[1, 2, 1],
            left=[5, 30, 10],
            top=[50, 10, 10],
            width=[20, 20, 15],
            height=[10, 10, 10],
            conf=[-1, 50, 100],
            text=["Later", "world", "Hello"],
        )
    )
    _install(monkeypatch, fake)
    backend = TesseractBackend()

    result = await backend.analyze(_backend_image(), OCRRequest())

    payloads = tuple(cast(TextPayload, item.payload) for item in result.observations)
    assert tuple(item.text for item in payloads) == ("Hello world", "Later")
    assert tuple(item.block_kind for item in payloads) == ("line", "line")
    assert result.observations[0].confidence == pytest.approx(0.75)
    assert result.observations[1].confidence is None
    assert result.observations[0].region == Box(
        x_min=0.1,
        y_min=0.1,
        x_max=0.5,
        y_max=0.2,
    )


@pytest.mark.asyncio
async def test_region_crop_maps_boxes_back_to_full_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract(data=_data(left=[0], top=[0], width=[25], height=[50], text=["region"]))
    _install(monkeypatch, fake)
    backend = TesseractBackend()
    region = Box(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)

    result = await backend.analyze(_backend_image(), OCRRequest(region=region))

    assert fake.analysis_arguments[0][0] == (50, 50)
    assert result.observations[0].region == Box(
        x_min=0.25,
        y_min=0.25,
        x_max=0.5,
        y_max=0.75,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    (("auto", "--psm 3"), ("sparse", "--psm 11"), ("dense", "--psm 6"), ("single_line", "--psm 7")),
)
async def test_mode_language_and_config_translation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: str,
) -> None:
    fake = FakePytesseract(languages=("eng", "deu"))
    _install(monkeypatch, fake)
    backend = TesseractBackend(languages=("eng", "deu"))
    request = OCRRequest.model_validate(
        {"languages": ("deu",), "mode": mode},
        strict=True,
    )

    result = await backend.analyze(_backend_image(), request)

    assert fake.analysis_arguments[0][1:3] == ("deu", expected)
    assert cast(TextPayload, result.observations[0].payload).language == "deu"


@pytest.mark.asyncio
async def test_explicit_config_and_executable_apply_to_every_runtime_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract()
    commands: list[str] = []
    original_version = fake.get_tesseract_version
    original_languages = fake.get_languages

    def version() -> str:
        commands.append(fake.pytesseract.tesseract_cmd)
        return original_version()

    def languages(config: str = "") -> list[str]:
        commands.append(fake.pytesseract.tesseract_cmd)
        assert config == "--oem 1 --psm 4"
        return original_languages(config=config)

    monkeypatch.setattr(fake, "get_tesseract_version", version)
    monkeypatch.setattr(fake, "get_languages", languages)
    _install(monkeypatch, fake)
    backend = TesseractBackend(
        executable="/configured/tesseract",
        config=" --oem 1 --psm 4 ",
    )

    await backend.analyze(_backend_image(), OCRRequest(mode="single_line"))

    assert commands == ["/configured/tesseract", "/configured/tesseract"]
    assert fake.analysis_arguments[0][2:] == (
        "--oem 1 --psm 4",
        fake.Output.DICT,
        "/configured/tesseract",
    )
    assert fake.pytesseract.tesseract_cmd == "system-tesseract"


@pytest.mark.asyncio
async def test_minimum_confidence_filters_after_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract(data=_data(conf=[49], text=["faint"]))
    _install(monkeypatch, fake)
    backend = TesseractBackend()

    result = await backend.analyze(
        _backend_image(),
        OCRRequest(min_confidence=0.5),
    )

    assert result.observations == ()
    assert tuple(item.code for item in result.warnings) == ("no_text_above_threshold",)


@pytest.mark.asyncio
async def test_concurrent_first_use_probes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract()
    _install(monkeypatch, fake)
    backend = TesseractBackend(max_concurrency=4)

    results = await asyncio.gather(
        *(backend.analyze(_backend_image(), OCRRequest()) for _ in range(6))
    )

    assert all(result.observations for result in results)
    assert fake.version_calls == 1
    assert fake.language_calls == 1
    assert fake.analysis_calls == 6


@pytest.mark.asyncio
async def test_cancelling_diagnostic_waiter_does_not_cancel_shared_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract()
    started = threading.Event()
    release = threading.Event()
    original = fake.get_tesseract_version

    def version() -> str:
        started.set()
        if not release.wait(2.0):
            raise TimeoutError
        return original()

    monkeypatch.setattr(fake, "get_tesseract_version", version)
    _install(monkeypatch, fake)
    backend = TesseractBackend()
    cancelled = asyncio.create_task(backend.analyze(_backend_image(), OCRRequest()))
    assert await asyncio.to_thread(started.wait, 1.0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    survivor = asyncio.create_task(backend.analyze(_backend_image(), OCRRequest()))
    release.set()
    result = await survivor

    assert result.observations
    assert fake.version_calls == 1


@pytest.mark.asyncio
async def test_close_waits_for_active_call_and_rejects_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePytesseract()
    started = threading.Event()
    release = threading.Event()
    original = fake.image_to_data

    def data(
        image: Image.Image,
        *,
        lang: str,
        config: str,
        output_type: object,
    ) -> Mapping[str, Sequence[object]]:
        started.set()
        if not release.wait(2.0):
            raise TimeoutError
        return original(image, lang=lang, config=config, output_type=output_type)

    monkeypatch.setattr(fake, "image_to_data", data)
    _install(monkeypatch, fake)
    backend = TesseractBackend()
    active = asyncio.create_task(backend.analyze(_backend_image(), OCRRequest()))
    assert await asyncio.to_thread(started.wait, 1.0)
    closing = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0)

    assert not closing.done()
    release.set()
    await active
    await closing
    await backend.aclose()
    with pytest.raises(SessionClosedError):
        await backend.analyze(_backend_image(), OCRRequest())


def test_language_and_configuration_selection_change_the_backend_version() -> None:
    baseline = TesseractBackend()
    other_languages = TesseractBackend(languages=("deu",))
    configured = TesseractBackend(config="--oem 1")
    other_config = TesseractBackend(config="--oem 0")

    assert baseline.descriptor.version == "1.0+lang.eng"
    assert other_languages.descriptor.version == "1.0+lang.deu"
    assert configured.descriptor.version != baseline.descriptor.version
    assert configured.descriptor.version != other_config.descriptor.version
    # An engine backend never reports an unresolved model revision.
    assert baseline.descriptor.model_id is None
    assert baseline.descriptor.model_revision is None
    assert baseline.descriptor.durable_cache_eligible is True
