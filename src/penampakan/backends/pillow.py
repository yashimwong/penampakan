"""Deterministic metadata and dominant-color perception with Pillow."""

from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image
from PIL import __version__ as pillow_version

from penampakan.image.geometry import box_to_pixels
from penampakan.models import (
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    ColorsPayload,
    ColorsRequest,
    ColorSwatch,
    MetadataPayload,
    MetadataRequest,
    ObservationDraft,
    VisionRequest,
    VisionResult,
    WarningInfo,
)

_BASIC_COLORS = {
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "brown": (128, 64, 0),
    "cyan": (0, 255, 255),
    "gray": (128, 128, 128),
    "green": (0, 128, 0),
    "magenta": (255, 0, 255),
    "orange": (255, 128, 0),
    "purple": (128, 0, 128),
    "red": (255, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
}


class PillowBackend:
    """Authoritative normalized metadata and deterministic dominant colors."""

    def __init__(self) -> None:
        self._descriptor = BackendDescriptor(
            name="penampakan.pillow",
            version=f"{pillow_version}+penampakan.1",
            capabilities=(
                CapabilityDescriptor(capability=Capability.METADATA),
                CapabilityDescriptor(capability=Capability.COLORS),
            ),
            max_concurrency=4,
        )
        self._closed = False

    @property
    def descriptor(self) -> BackendDescriptor:
        """Return stable built-in backend metadata."""

        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        """Return whether the request is metadata or dominant colors."""

        return isinstance(request, (MetadataRequest, ColorsRequest))

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        """Analyze canonical pixels without blocking the event loop."""

        if self._closed:
            raise RuntimeError("backend is closed")
        if not self.supports(request):
            raise ValueError("request is unsupported")
        if isinstance(request, MetadataRequest):
            payload = MetadataPayload(
                width=image.asset.width,
                height=image.asset.height,
                aspect_ratio=image.asset.width / image.asset.height,
                has_alpha=image.asset.mode == "RGBA",
            )
            return VisionResult(observations=(ObservationDraft(payload=payload),))
        if not isinstance(request, ColorsRequest):
            raise ValueError("request is unsupported")
        return await asyncio.to_thread(self._extract_colors, image, request)

    async def aclose(self) -> None:
        """Mark the stateless backend closed idempotently."""

        self._closed = True

    def _extract_colors(self, backend_image: BackendImage, request: ColorsRequest) -> VisionResult:
        with Image.open(BytesIO(backend_image.content)) as decoded:
            decoded.load()
            working = decoded.copy()
        try:
            if request.region is not None:
                bounds = box_to_pixels(
                    request.region,
                    backend_image.asset.width,
                    backend_image.asset.height,
                )
                cropped = working.crop(bounds.as_tuple())
                working.close()
                working = cropped
            composited, warning = _composite_transparency(working)
            if composited is not working:
                working.close()
                working = composited
            sampled = working.copy()
            sampled.thumbnail((256, 256), Image.Resampling.LANCZOS)
            if sampled.mode != "RGB":
                converted = sampled.convert("RGB")
                sampled.close()
                sampled = converted
            try:
                swatches = _quantized_swatches(sampled, request.count)
            finally:
                sampled.close()
            warnings = () if warning is None else (warning,)
            return VisionResult(
                observations=(
                    ObservationDraft(
                        payload=ColorsPayload(swatches=swatches),
                        region=request.region,
                    ),
                ),
                warnings=warnings,
            )
        finally:
            working.close()


def _composite_transparency(image: Image.Image) -> tuple[Image.Image, WarningInfo | None]:
    if image.mode != "RGBA":
        return image, None
    extrema = image.getchannel("A").getextrema()
    if extrema is None or extrema[0] == 255:
        converted = image.convert("RGB")
        return converted, None
    background = Image.new("RGB", image.size)
    pixels = background.load()
    if pixels is None:
        background.close()
        raise RuntimeError("unable to access checkerboard pixels")
    for y in range(image.height):
        for x in range(image.width):
            shade = 192 if (x // 8 + y // 8) % 2 == 0 else 128
            pixels[x, y] = (shade, shade, shade)
    alpha = image.getchannel("A")
    try:
        background.paste(image, mask=alpha)
    finally:
        alpha.close()
    return background, WarningInfo(
        code="transparent_color_estimate",
        message="Dominant colors were estimated over a neutral checkerboard.",
    )


def _quantized_swatches(image: Image.Image, count: int) -> tuple[ColorSwatch, ...]:
    quantized = image.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    try:
        palette = quantized.getpalette()
        counts = quantized.getcolors(maxcolors=image.width * image.height)
        if palette is None or counts is None:
            return ()
        total = sum(amount for amount, _ in counts)
        colors: list[tuple[int, tuple[int, int, int]]] = []
        for amount, index in counts:
            if not isinstance(index, int):
                continue
            offset = index * 3
            rgb = tuple(palette[offset : offset + 3])
            if len(rgb) == 3:
                colors.append((amount, (rgb[0], rgb[1], rgb[2])))
        colors.sort(key=lambda item: (-item[0], item[1]))
        fractions = [amount / total for amount, _ in colors]
        if fractions:
            fractions[-1] += 1.0 - sum(fractions)
        return tuple(
            ColorSwatch(
                rgb=rgb,
                hex=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                fraction=fraction,
                name=_nearest_color_name(rgb),
            )
            for (_, rgb), fraction in zip(colors, fractions, strict=True)
        )
    finally:
        quantized.close()


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    return min(
        _BASIC_COLORS,
        key=lambda name: sum(
            (component - reference) ** 2
            for component, reference in zip(rgb, _BASIC_COLORS[name], strict=True)
        ),
    )


__all__ = ["PillowBackend"]
