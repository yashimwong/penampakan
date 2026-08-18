#!/usr/bin/env python3
"""Validate local links and GitHub-style anchors in tracked Markdown files.

The default, deterministic mode never accesses the network.  ``--external-only``
is intended for the separate, non-blocking scheduled CI job.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# There are currently no generated-at-publish Markdown targets. Add exact,
# repository-relative paths here only when a publishing step intentionally
# creates a documented target after this check runs.
GENERATED_PATH_ALLOWLIST: frozenset[str] = frozenset()

EXTERNAL_SCHEMES = frozenset({"http", "https"})
IGNORED_SCHEMES = frozenset({"mailto", "tel", "data"})
PLACEHOLDER_MARKERS = (
    "example.com",
    "github.com/owner/",
    "github.com/your-",
    "github.com/your_",
    "readthedocs.io/projects/your-",
)
_REFERENCE_DEFINITION = re.compile(r"^ {0,3}\[([^]]+)\]:\s*(?:<([^>]+)>|(\S+))", re.MULTILINE)
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$", re.MULTILINE)
_SETEXT_HEADING = re.compile(r"^ {0,3}(.+?)\s*\n {0,3}(=+|-+)\s*$", re.MULTILINE)
_HTML_ANCHOR = re.compile(r"<(?:a|[A-Za-z][\w:-]*)\s+[^>]*(?:id|name)=[\"']([^\"']+)[\"']", re.I)


@dataclass(frozen=True)
class MarkdownLink:
    target: str
    line: int


def _without_fenced_code(text: str) -> str:
    """Blank fenced code while preserving line numbers."""

    output: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                fence = None
            output.append("\n" if line.endswith("\n") else "")
        elif fence is None:
            output.append(line)
        else:
            output.append("\n" if line.endswith("\n") else "")
    return "".join(output)


def _without_code(text: str) -> str:
    text = _without_fenced_code(text)
    return re.sub(
        r"(`+)(.+?)\1",
        lambda match: "".join("\n" if char == "\n" else " " for char in match.group(0)),
        text,
        flags=re.DOTALL,
    )


def _normalize_reference(label: str) -> str:
    return " ".join(label.split()).casefold()


def _destination(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<"):
        end = raw.find(">", 1)
        return raw[1:end] if end >= 0 else raw[1:]

    depth = 0
    escaped = False
    for index, character in enumerate(raw):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        elif character.isspace() and depth == 0:
            return raw[:index]
    return raw


def _closing(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 1
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def markdown_links(text: str) -> tuple[MarkdownLink, ...]:
    """Extract inline, image, and full/collapsed/shortcut reference links."""

    text = _without_code(text)
    definitions: dict[str, str] = {}
    definition_spans: list[tuple[int, int]] = []
    for match in _REFERENCE_DEFINITION.finditer(text):
        definitions.setdefault(
            _normalize_reference(match.group(1)), match.group(2) or match.group(3)
        )
        definition_spans.append(match.span())

    links: list[MarkdownLink] = []
    index = 0
    while index < len(text):
        if text[index] != "[" or (index and text[index - 1] == "\\"):
            index += 1
            continue
        if any(start <= index < end for start, end in definition_spans):
            index += 1
            continue

        label_end = _closing(text, index + 1, "[", "]")
        if label_end is None:
            index += 1
            continue
        label = text[index + 1 : label_end]
        cursor = label_end + 1
        line = text.count("\n", 0, index) + 1
        if cursor < len(text) and text[cursor] == "(":
            target_end = _closing(text, cursor + 1, "(", ")")
            if target_end is not None:
                links.append(MarkdownLink(_destination(text[cursor + 1 : target_end]), line))
                # Continue inside the label too: linked images such as
                # ``[![alt](image.png)](page.md)`` contain two targets.
                index += 1
                continue
        elif cursor < len(text) and text[cursor] == "[":
            reference_end = _closing(text, cursor + 1, "[", "]")
            if reference_end is not None:
                reference = text[cursor + 1 : reference_end] or label
                target = definitions.get(_normalize_reference(reference))
                if target is not None:
                    links.append(MarkdownLink(target, line))
                index = reference_end + 1
                continue
        else:
            target = definitions.get(_normalize_reference(label))
            if target is not None:
                links.append(MarkdownLink(target, line))
        index = label_end + 1
    return tuple(links)


def undefined_references(text: str) -> tuple[MarkdownLink, ...]:
    """Return explicit reference links whose definitions are absent."""

    text = _without_code(text)
    definitions = {
        _normalize_reference(match.group(1)) for match in _REFERENCE_DEFINITION.finditer(text)
    }
    missing: list[MarkdownLink] = []
    for match in re.finditer(r"\[([^]\n]+)\]\[([^]\n]*)\]", text):
        reference = match.group(2) or match.group(1)
        if _normalize_reference(reference) not in definitions:
            missing.append(MarkdownLink(reference, text.count("\n", 0, match.start()) + 1))
    return tuple(missing)


def _heading_text(value: str) -> str:
    value = re.sub(r"[ \t]+#+[ \t]*$", "", value)
    value = re.sub(r"!?(?:\[([^]]*)\])(?:\([^)]*\)|\[[^]]*\])", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"[`*_~]", "", html.unescape(value)).strip()


def github_slug(value: str) -> str:
    """Return the base heading slug produced by GitHub's Markdown renderer."""

    value = _heading_text(value).lower().strip()
    value = re.sub(r"[^\w\- ]", "", value, flags=re.UNICODE)
    return re.sub(r"\s+", "-", value)


def markdown_anchors(text: str) -> frozenset[str]:
    text = _without_fenced_code(text)
    headings: list[tuple[int, str]] = []
    for match in _ATX_HEADING.finditer(text):
        headings.append((match.start(), match.group(2) or ""))
    for match in _SETEXT_HEADING.finditer(text):
        # A line already parsed as an ATX heading cannot also be setext content.
        if not match.group(1).lstrip().startswith("#"):
            headings.append((match.start(), match.group(1)))

    anchors: set[str] = {
        urllib.parse.unquote(match.group(1)) for match in _HTML_ANCHOR.finditer(text)
    }
    used_headings: set[str] = set()
    counts: defaultdict[str, int] = defaultdict(int)
    for _, heading in sorted(headings):
        base = github_slug(heading)
        slug = base
        while slug in used_headings:
            counts[base] += 1
            slug = f"{base}-{counts[base]}"
        used_headings.add(slug)
        anchors.add(slug)
    return frozenset(anchors)


def _tracked_markdown(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md", "*.markdown"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(root / item.decode() for item in result.stdout.split(b"\0") if item)


def _has_exact_case(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        if part == ".." or not current.is_dir():
            return False
        try:
            names = {entry.name for entry in current.iterdir()}
        except OSError:
            return False
        if part not in names:
            return False
        current /= part
    return current.exists()


def _local_target(root: Path, document: Path, path: str) -> tuple[Path, Path] | None:
    decoded = urllib.parse.unquote(path)
    if not decoded:
        return document, document.relative_to(root)
    base_parts: list[str] = (
        [] if decoded.startswith("/") else list(document.parent.relative_to(root).parts)
    )
    for part in Path(decoded.lstrip("/")).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not base_parts:
                return None
            base_parts.pop()
        else:
            base_parts.append(part)
    normalized = Path(*base_parts)
    target = root / normalized
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target, normalized


def validate_markdown(root: Path, documents: Iterable[Path]) -> list[str]:
    """Return deterministic local-link errors for ``documents``."""

    root = root.resolve()
    errors: list[str] = []
    anchor_cache: dict[Path, frozenset[str]] = {}
    for document in sorted(Path(item).resolve() for item in documents):
        display = document.relative_to(root).as_posix()
        text = document.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered_line = line.casefold()
            if any(marker in lowered_line for marker in PLACEHOLDER_MARKERS):
                errors.append(f"{display}:{line_number}: placeholder URL")
        for reference in undefined_references(text):
            errors.append(
                f"{display}:{reference.line}: undefined link reference: {reference.target}"
            )
        for link in markdown_links(text):
            target_text = html.unescape(link.target).strip()
            parsed = urllib.parse.urlsplit(target_text)
            if parsed.scheme.casefold() in EXTERNAL_SCHEMES or target_text.startswith("//"):
                continue
            if parsed.scheme.casefold() in IGNORED_SCHEMES:
                continue
            if parsed.scheme or parsed.netloc:
                errors.append(f"{display}:{link.line}: unsupported link scheme: {target_text}")
                continue

            local = _local_target(root, document, parsed.path)
            if local is None:
                errors.append(f"{display}:{link.line}: target escapes repository: {target_text}")
                continue
            target, relative = local
            relative_name = relative.as_posix()
            if relative_name not in GENERATED_PATH_ALLOWLIST and not _has_exact_case(
                root, relative
            ):
                errors.append(
                    f"{display}:{link.line}: missing or case-mismatched target: {target_text}"
                )
                continue

            fragment = urllib.parse.unquote(parsed.fragment)
            if fragment and target.is_file() and target.suffix.casefold() in {".md", ".markdown"}:
                anchors = anchor_cache.setdefault(
                    target, markdown_anchors(target.read_text(encoding="utf-8"))
                )
                if fragment not in anchors:
                    errors.append(
                        f"{display}:{link.line}: missing anchor #{fragment}: {target_text}"
                    )
    return errors


def _external_urls(documents: Iterable[Path]) -> tuple[str, ...]:
    urls: set[str] = set()
    for document in documents:
        for link in markdown_links(document.read_text(encoding="utf-8")):
            if urllib.parse.urlsplit(link.target).scheme.casefold() in EXTERNAL_SCHEMES:
                urls.add(html.unescape(link.target))
    return tuple(sorted(urls))


def _check_external(url: str, timeout: float) -> str | None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "penampakan-link-check/1", "Range": "bytes=0-1023"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 400:
                return f"external link returned HTTP {response.status}: {url}"
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return f"external link failed ({error}): {url}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files (default: tracked)")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--external-only", action="store_true", help="check remote URLs")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-request timeout")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    documents = tuple(
        path if path.is_absolute() else root / path for path in arguments.paths
    ) or _tracked_markdown(root)

    if arguments.external_only:
        with ThreadPoolExecutor(max_workers=8) as executor:
            errors = tuple(
                error
                for error in executor.map(
                    lambda url: _check_external(url, arguments.timeout), _external_urls(documents)
                )
                if error is not None
            )
    else:
        errors = tuple(validate_markdown(root, documents))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"checked {len(documents)} Markdown file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
