from importlib.metadata import metadata, requires
from importlib.resources import files


def test_distribution_metadata_declares_supported_runtime() -> None:
    package_metadata = metadata("penampakan")

    assert package_metadata["Requires-Python"] == ">=3.10,<3.14"
    assert package_metadata["License"] == "MIT"


def test_distribution_metadata_declares_dependencies_and_extras() -> None:
    dependencies = tuple(requires("penampakan") or ())

    assert any(item.startswith("Pillow>=10") for item in dependencies)
    assert any(item.startswith("pydantic<3,>=2.7") for item in dependencies)
    assert any('extra == "ocr"' in item for item in dependencies)
    assert any('extra == "transformers"' in item for item in dependencies)
    assert any('extra == "dev"' in item for item in dependencies)


def test_distribution_contains_typing_marker() -> None:
    marker = files("penampakan").joinpath("py.typed")

    assert marker.is_file()
