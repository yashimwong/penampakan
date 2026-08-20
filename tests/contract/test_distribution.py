from importlib.metadata import metadata, requires
from importlib.resources import files

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def test_distribution_metadata_declares_supported_runtime() -> None:
    package_metadata = metadata("penampakan")

    assert SpecifierSet(package_metadata["Requires-Python"]) == SpecifierSet(">=3.10,<3.14")
    assert "MIT" in package_metadata["License"]
    assert package_metadata.get_all("Classifier") is not None
    assert "License :: OSI Approved :: MIT License" in package_metadata.get_all("Classifier")
    assert "Typing :: Typed" in package_metadata.get_all("Classifier")


def test_distribution_metadata_uses_canonical_project_urls() -> None:
    package_metadata = metadata("penampakan")
    project_urls = {
        label.strip(): url.strip()
        for item in package_metadata.get_all("Project-URL") or ()
        for label, separator, url in [item.partition(",")]
        if separator
    }

    assert project_urls == {
        "Homepage": "https://github.com/yashimwong/penampakan",
        "Repository": "https://github.com/yashimwong/penampakan.git",
        "Documentation": "https://github.com/yashimwong/penampakan/tree/master/docs",
        "Changelog": "https://github.com/yashimwong/penampakan/blob/master/CHANGELOG.md",
        "Issues": "https://github.com/yashimwong/penampakan/issues",
    }
    assert all(
        url.startswith("https://github.com/yashimwong/penampakan") for url in project_urls.values()
    )
    assert not any("placeholder" in url or "example.com" in url for url in project_urls.values())


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
    assert set(metadata("penampakan").get_all("Provides-Extra") or ()) == {
        "anthropic",
        "benchmark",
        "dev",
        "litellm",
        "ocr",
        "openai",
        "opentelemetry",
        "providers",
        "transformers",
    }


def test_distribution_metadata_declares_bounded_provider_extras() -> None:
    dependencies = tuple(Requirement(item) for item in requires("penampakan") or ())
    expected = {
        "openai": ("openai", SpecifierSet(">=2.54,<3")),
        "anthropic": ("anthropic", SpecifierSet(">=0.121,<1")),
        "litellm": ("litellm", SpecifierSet(">=1.96.2,<2")),
    }

    for extra, (name, specifier) in expected.items():
        matches = [
            item
            for item in dependencies
            if item.name.lower() == name
            and item.marker is not None
            and f'extra == "{extra}"' in str(item.marker)
        ]
        assert matches, extra
        # A pinned upper major bound is required because no provider SDK
        # promises compatibility across a major release.
        assert matches[0].specifier == specifier

    assert any(
        item.marker is not None and 'extra == "providers"' in str(item.marker)
        for item in dependencies
    )


def test_distribution_metadata_pins_opentelemetry_extra() -> None:
    dependencies = tuple(Requirement(item) for item in requires("penampakan") or ())
    expected = {
        "opentelemetry-api": SpecifierSet("==1.44.0"),
        "opentelemetry-sdk": SpecifierSet("==1.44.0"),
        "opentelemetry-semantic-conventions": SpecifierSet("==0.65b0"),
    }

    selected = {
        item.name.lower(): item.specifier
        for item in dependencies
        if item.marker is not None and 'extra == "opentelemetry"' in str(item.marker)
    }
    assert selected == expected


def test_distribution_contains_typing_marker() -> None:
    marker = files("penampakan").joinpath("py.typed")

    assert marker.is_file()
