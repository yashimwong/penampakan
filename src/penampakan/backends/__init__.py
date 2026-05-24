"""Optional and built-in vision backend adapters."""

from penampakan.backends.callable import CallableVisionBackend
from penampakan.backends.pillow import PillowBackend

__all__ = ("CallableVisionBackend", "PillowBackend")
