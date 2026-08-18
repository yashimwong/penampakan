from pathlib import Path

from scripts.check_links import (
    github_slug,
    markdown_anchors,
    markdown_links,
    undefined_references,
    validate_markdown,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_link_parser_handles_images_references_and_code_fences() -> None:
    text = """\
![inline](images/a.png)
[reference][diagram]
[collapsed][]
[shortcut]
[![nested](images/nested.png)](guide.md)
`[not a link](missing.md)`
```markdown
[also not a link](missing.md)
```

[diagram]: images/diagram.png "title"
[collapsed]: <a path/file.md>
[shortcut]: target.md
"""

    assert [link.target for link in markdown_links(text)] == [
        "images/a.png",
        "images/diagram.png",
        "a path/file.md",
        "target.md",
        "guide.md",
        "images/nested.png",
    ]


def test_github_anchors_include_duplicate_and_url_decoded_fragments(tmp_path: Path) -> None:
    document = _write(
        tmp_path / "docs" / "guide.md",
        '# Café & setup\n\n## Repeat\n\n## Repeat\n\n<a id="manual"></a>\n',
    )
    source = _write(
        tmp_path / "README.md",
        "[unicode](docs/guide.md#caf%C3%A9-setup)\n"
        "[duplicate](docs/guide.md#repeat-1)\n"
        "[explicit](/docs/guide.md#manual)\n",
    )

    assert github_slug("Café & setup") == "café-setup"
    anchors = markdown_anchors(document.read_text(encoding="utf-8"))

    assert anchors >= {"café-setup", "repeat", "repeat-1", "manual"}
    assert validate_markdown(tmp_path, [source]) == []


def test_local_link_validation_reports_case_missing_escape_anchor_and_placeholder(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "docs" / "Target File.md", "# Existing heading\n")
    source = _write(
        tmp_path / "docs" / "source.md",
        "[case](target%20File.md)\n"
        "[missing](absent.md)\n"
        "[escape](../../outside.md)\n"
        "[anchor](Target%20File.md#not-there)\n"
        "[placeholder](https://example.com/project)\n",
    )

    errors = validate_markdown(tmp_path, [source])

    assert len(errors) == 5
    assert any("case-mismatched" in error for error in errors)
    assert any("missing or case-mismatched" in error and "absent.md" in error for error in errors)
    assert any("escapes repository" in error for error in errors)
    assert any("missing anchor" in error for error in errors)
    assert any("placeholder URL" in error for error in errors)


def test_fragment_only_link_and_document_relative_parent_are_valid(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "docs" / "nested" / "source.md",
        "# Here\n\n[self](#here)\n[parent](../target.md#target)\n",
    )
    _write(tmp_path / "docs" / "target.md", "# Target\n")

    assert validate_markdown(tmp_path, [source]) == []


def test_undefined_full_and_collapsed_references_are_reported(tmp_path: Path) -> None:
    text = "[missing][no definition]\n[also missing][]\n"
    source = _write(tmp_path / "README.md", text)

    assert [item.target for item in undefined_references(text)] == ["no definition", "also missing"]
    assert len(validate_markdown(tmp_path, [source])) == 2


def test_bare_placeholder_url_is_reported(tmp_path: Path) -> None:
    source = _write(tmp_path / "README.md", "Configure https://github.com/owner/project first.\n")

    assert validate_markdown(tmp_path, [source]) == ["README.md:1: placeholder URL"]
