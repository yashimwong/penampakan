"""Optional and built-in vision backend adapters."""

from penampakan.backends.callable import CallableVisionBackend
from penampakan.backends.pillow import PillowBackend
from penampakan.backends.tesseract import TesseractBackend
from penampakan.backends.transformers import (
    TransformersCaptionBackend,
    TransformersDetectionBackend,
)

__all__ = (
    "CallableVisionBackend",
    "PillowBackend",
    "TesseractBackend",
    "TransformersCaptionBackend",
    "TransformersDetectionBackend",
)
