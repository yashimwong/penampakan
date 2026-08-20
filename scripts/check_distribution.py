#!/usr/bin/env python3
"""Inspect built wheel and sdist contents and core package metadata."""

from __future__ import annotations

import argparse
import email.parser
import re
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

REQUIRED_URL_LABELS = frozenset({"Homepage", "Repository", "Documentation", "Changelog", "Issues"})
PLACEHOLDER_MARKERS = ("example.com", "github.com/owner/", "your-org", "your_username")


def _wheel_files(path: Path) -> tuple[set[str], bytes]:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"expected one wheel METADATA file, found {len(metadata_names)}")
        return names, archive.read(metadata_names[0])


def _sdist_files(path: Path) -> tuple[set[str], bytes]:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        metadata_names = [name for name in members if name.endswith("/PKG-INFO")]
        if len(metadata_names) != 1:
            raise ValueError(f"expected one sdist PKG-INFO file, found {len(metadata_names)}")
        stream = archive.extractfile(members[metadata_names[0]])
        if stream is None:
            raise ValueError("could not read sdist PKG-INFO")
        return set(members), stream.read()


def _metadata_errors(raw: bytes, artifact: Path) -> list[str]:
    metadata = email.parser.BytesParser().parsebytes(raw)
    errors: list[str] = []
    if metadata.get("Name") != "penampakan":
        errors.append(f"{artifact}: unexpected package name {metadata.get('Name')!r}")
    python_constraints = {item.strip() for item in metadata.get("Requires-Python", "").split(",")}
    if python_constraints != {">=3.10", "<3.14"}:
        errors.append(f"{artifact}: unexpected Requires-Python")
    classifiers = set(metadata.get_all("Classifier", ()))
    for minor in range(10, 14):
        classifier = f"Programming Language :: Python :: 3.{minor}"
        if classifier not in classifiers:
            errors.append(f"{artifact}: missing classifier {classifier!r}")
    if "Typing :: Typed" not in classifiers:
        errors.append(f"{artifact}: missing typed-package classifier")
    if "License :: OSI Approved :: MIT License" not in classifiers:
        errors.append(f"{artifact}: missing MIT classifier")

    urls: dict[str, str] = {}
    for value in metadata.get_all("Project-URL", ()):
        label, separator, url = value.partition(",")
        if separator:
            urls[label.strip()] = url.strip()
    missing = REQUIRED_URL_LABELS - urls.keys()
    if missing:
        errors.append(f"{artifact}: missing project URLs: {', '.join(sorted(missing))}")
    for label, url in urls.items():
        lowered = url.casefold()
        if not re.match(r"^https://github\.com/yashimwong/penampakan(?:[./]|$)", url):
            errors.append(f"{artifact}: {label} is not a canonical project URL: {url}")
        if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
            errors.append(f"{artifact}: {label} contains a placeholder: {url}")

    extras = set(metadata.get_all("Provides-Extra", ()))
    expected_extras = {
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
    if extras != expected_extras:
        errors.append(
            f"{artifact}: extras differ: expected {sorted(expected_extras)}, got {sorted(extras)}"
        )
    return errors


def inspect_artifacts(paths: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    wheels = [path for path in paths if path.suffix == ".whl"]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        return [f"expected one wheel and one .tar.gz sdist, got {len(wheels)} and {len(sdists)}"]

    for path, reader in ((wheels[0], _wheel_files), (sdists[0], _sdist_files)):
        try:
            names, metadata = reader(path)
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            errors.append(f"{path}: invalid artifact: {error}")
            continue
        errors.extend(_metadata_errors(metadata, path))
        if not any(
            name.endswith("/penampakan/py.typed") or name == "penampakan/py.typed" for name in names
        ):
            errors.append(f"{path}: missing penampakan/py.typed")
        if not any(
            name.endswith("/LICENSE") or ".dist-info/licenses/LICENSE" in name for name in names
        ):
            errors.append(f"{path}: missing LICENSE")
        if path.name.endswith(".tar.gz"):
            for required in ("README.md", "CHANGELOG.md", "pyproject.toml"):
                if not any(name.endswith(f"/{required}") for name in names):
                    errors.append(f"{path}: missing {required}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    arguments = parser.parse_args(argv)
    errors = inspect_artifacts(arguments.artifacts)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("wheel and sdist metadata/content checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
