import importlib
import sys


def test_import_is_lightweight() -> None:
    optional_modules = {"torch", "transformers", "pytesseract"}
    for name in optional_modules:
        sys.modules.pop(name, None)

    module = importlib.import_module("penampakan")

    assert module.__version__ == "0.1.0"
    assert optional_modules.isdisjoint(sys.modules)


def test_public_exports_are_curated() -> None:
    module = importlib.import_module("penampakan")

    assert module.__all__ == ("__version__",)
