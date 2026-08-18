import importlib
import sys


def test_import_is_lightweight() -> None:
    optional_modules = {"torch", "transformers", "pytesseract"}
    for name in optional_modules:
        sys.modules.pop(name, None)

    module = importlib.import_module("penampakan")

    assert module.__version__ == "0.4.0"
    assert optional_modules.isdisjoint(sys.modules)


def test_backend_exports_are_lightweight() -> None:
    optional_modules = {"torch", "transformers", "pytesseract"}
    for name in optional_modules:
        sys.modules.pop(name, None)

    module = importlib.import_module("penampakan.backends")

    # Specification 05 D2: tests compare required inclusion, not equality, so a
    # new backend export never breaks this gate.
    expected = {
        "CallableVisionBackend",
        "PillowBackend",
        "TesseractBackend",
        "TransformersCaptionBackend",
        "TransformersDetectionBackend",
    }

    assert expected <= set(module.__all__)
    assert all(hasattr(module, name) for name in module.__all__)
    assert optional_modules.isdisjoint(sys.modules)


def test_public_exports_are_curated() -> None:
    module = importlib.import_module("penampakan")

    expected = {
        "ActionPolicy",
        "AsyncPenampakan",
        "AsyncVisionSession",
        "CallableTextLLM",
        "CallableVisionBackend",
        "Capability",
        "InspectionPlan",
        "InspectionResult",
        "Penampakan",
        "PenampakanError",
        "PillowBackend",
        "Settings",
        "TextLLM",
        "VisionAnswer",
        "VisionBackend",
        "VisionSession",
        "__version__",
    }

    assert expected <= set(module.__all__)
    assert len(module.__all__) == len(set(module.__all__))
    assert all(hasattr(module, name) for name in module.__all__)
