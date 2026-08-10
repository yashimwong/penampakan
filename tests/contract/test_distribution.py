from importlib.metadata import metadata, requires
from importlib.resources import files

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def test_distribution_metadata_declares_supported_runtime() -> None:
    package_metadata = metadata("penampakan")

    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.14")
    assert package_metadata["License"] == "MIT"


def test_distribution_metadata_declares_dependencies_and_extras() -> None:
    dependencies = tuple(Requirement(item) for item in requires("penampakan") or ())

    assert any(
        item.name.lower() == "pillow" and item.specifier == SpecifierSet(">=10")
        for item in dependencies
    )
    assert any(
        item.name.lower() == "pydantic" and item.specifier == SpecifierSet(">=2.7,<3")
        for item in dependencies
    )
    assert any(
        item.marker is not None and 'extra == "ocr"' in str(item.marker) for item in dependencies
    )
    assert any(
        item.marker is not None and 'extra == "transformers"' in str(item.marker)
        for item in dependencies
    )
    assert any(
        item.marker is not None and 'extra == "benchmark"' in str(item.marker)
        for item in dependencies
    )
    assert any(
        item.marker is not None and 'extra == "dev"' in str(item.marker) for item in dependencies
    )


def test_distribution_contains_typing_marker() -> None:
    marker = files("penampakan").joinpath("py.typed")

    assert marker.is_file()
