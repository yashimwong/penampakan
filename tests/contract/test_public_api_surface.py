"""Contract tests for the tiered public API surface of specification 05 D5.

The cases here cover the D5 bullets that no other contract module owns: README
import paths, the additive top-level helpers, the documented tier-2 namespaces,
star/``dir``/``help`` behaviour on a base install, ``AttributeError`` semantics,
the frozen export snapshot, and precise type-checker resolution for the optional
adapter classes.

Every base-install case runs in a subprocess whose import machinery hides all
optional third-party packages, so the results hold even when those packages are
installed for the real-dependency integration tests.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import penampakan
from penampakan.errors import ConfigurationError

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "README.md"
_SNAPSHOT = Path(__file__).with_name("public_api.json")

_OPTIONAL_PACKAGES = (
    "anthropic",
    "huggingface_hub",
    "litellm",
    "openai",
    "pytesseract",
    "torch",
    "transformers",
)

# Documented tier-2 namespaces. ``penampakan.tracing.sinks`` is deliberately
# absent: specification 08 owns it and it has not shipped.
_NAMESPACES = (
    "penampakan.backends",
    "penampakan.evaluation",
    "penampakan.image",
    "penampakan.llms",
    "penampakan.perception",
    "penampakan.perception.cache",
    "penampakan.reasoning",
    "penampakan.tools",
    "penampakan.tracing",
)

_BLOCKER = """
import sys


class _BlockOptional:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {blocked!r}:
            raise ImportError(f"no module named {{root}}")
        return None


for _name in list(sys.modules):
    if _name.split(".")[0] in {blocked!r}:
        del sys.modules[_name]
sys.meta_path.insert(0, _BlockOptional())
"""

_IMPORT_STATEMENT = re.compile(
    r"^(?:from\s+[.\w]+\s+import\s+(?:\([^()]*\)|[^\n(]+)|import\s+[^\n]+)",
    re.MULTILINE,
)


def _run(body: str) -> str:
    script = _BLOCKER.format(blocked=set(_OPTIONAL_PACKAGES)) + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _readme_imports() -> tuple[tuple[str, str | None], ...]:
    """Return every ``(module, name)`` pair imported from ``penampakan`` in README code."""
    text = _README.read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
    assert len(blocks) >= 4, "no fenced python blocks were extracted from the README"

    imports: list[tuple[str, str | None]] = []
    for block in blocks:
        for match in _IMPORT_STATEMENT.finditer(textwrap.dedent(block)):
            statement = ast.parse(match.group(0))
            for node in statement.body:
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module.split(".")[0] != "penampakan":
                        continue
                    imports.extend((module, alias.name) for alias in node.names)
                elif isinstance(node, ast.Import):
                    imports.extend(
                        (alias.name, None)
                        for alias in node.names
                        if alias.name.split(".")[0] == "penampakan"
                    )
    return tuple(imports)


def test_the_readme_import_extraction_is_not_silently_empty() -> None:
    imports = _readme_imports()
    modules = {module for module, _ in imports}

    # Guards against a regex that stops matching and turns the parametrized
    # resolution test below into a vacuous pass.
    assert len(imports) >= 10, imports
    assert len(modules) >= 3, modules


@pytest.mark.parametrize(("module", "name"), _readme_imports())
def test_every_readme_import_resolves_from_the_path_shown(module: str, name: str | None) -> None:
    imported = importlib.import_module(module)

    if name is not None:
        assert hasattr(imported, name), f"README imports {name} from {module}"


def test_every_optional_package_is_absent_in_the_probe_environment() -> None:
    installed = [name for name in _OPTIONAL_PACKAGES if importlib.util.find_spec(name) is not None]
    if not installed:
        # A base install has none of them, which is exactly the environment this
        # specification targets: the blocker is trivially satisfied and there is
        # nothing to prove about it here.
        pytest.skip("no optional package is installed, so the blocker cannot be proven non-vacuous")

    output = _run(
        f"""
        import importlib

        for name in {_OPTIONAL_PACKAGES!r}:
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            raise AssertionError(name)
        print("blocked")
        """
    )
    assert output.strip() == "blocked"


def test_the_three_additive_top_level_helpers_import_on_a_base_install() -> None:
    output = _run(
        """
        import inspect

        from penampakan import CallableTextLLM, CallableVisionBackend, PillowBackend

        for helper in (CallableTextLLM, CallableVisionBackend, PillowBackend):
            assert inspect.isclass(helper), helper
        print(",".join(helper.__name__ for helper in (
            CallableTextLLM,
            CallableVisionBackend,
            PillowBackend,
        )))
        """
    )
    assert output.strip() == "CallableTextLLM,CallableVisionBackend,PillowBackend"


@pytest.mark.parametrize("module", _NAMESPACES)
def test_documented_namespaces_import_and_declare_an_intentional_all(module: str) -> None:
    imported = importlib.import_module(module)

    exported = getattr(imported, "__all__", None)
    assert exported is not None, f"{module} declares no __all__"
    assert isinstance(exported, tuple | list), f"{module}.__all__ is {type(exported)!r}"
    assert all(isinstance(name, str) for name in exported), module
    assert len(set(exported)) == len(exported), f"{module}.__all__ repeats a name"
    for name in exported:
        assert hasattr(imported, name), f"{module}.__all__ names missing {name}"


def test_base_and_star_import_work_without_any_optional_package() -> None:
    output = _run(
        """
        import contextlib
        import io
        import pydoc

        import penampakan

        namespace = {}
        exec("from penampakan import *", namespace)
        for name in penampakan.__all__:
            assert hasattr(penampakan, name), name
            assert name in namespace, name
        assert "Penampakan" in dir(penampakan)
        assert "Penampakan" in pydoc.render_doc("penampakan")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            help(penampakan)
        assert "Penampakan" in buffer.getvalue()
        print("ok")
        """
    )
    assert output.strip() == "ok"


@pytest.mark.parametrize("module", ("penampakan", "penampakan.backends", "penampakan.llms"))
def test_dir_includes_the_exports_without_hiding_ordinary_attributes(module: str) -> None:
    imported = importlib.import_module(module)
    listing = dir(imported)

    # Specification 05 D2: required inclusion, never exact equality.
    assert set(imported.__all__) <= set(listing)
    ordinary = {
        name for name in listing if not name.startswith("__") and name not in imported.__all__
    }
    assert ordinary, f"dir({module}) pretends the module has no ordinary non-dunder attributes"


def test_unknown_attributes_raise_a_normal_attribute_error() -> None:
    unknown = "NoSuchName"

    with pytest.raises(AttributeError) as caught:
        getattr(penampakan, unknown)

    assert not isinstance(caught.value, ConfigurationError)


def test_unknown_attributes_raise_attribute_error_without_optional_packages() -> None:
    output = _run(
        """
        import penampakan
        from penampakan.errors import ConfigurationError

        for module in (penampakan, penampakan.backends, penampakan.llms):
            try:
                getattr(module, "NoSuchName")
            except ConfigurationError as error:
                raise AssertionError(f"{module.__name__}: {error!r}") from error
            except AttributeError:
                continue
            raise AssertionError(module.__name__)
        print("attribute-error")
        """
    )
    assert output.strip() == "attribute-error"


def test_every_snapshotted_export_is_still_present() -> None:
    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    names = snapshot["names"]

    assert names == sorted(names)
    assert len(set(names)) == len(names)
    assert snapshot["description"].strip()
    # Inclusion, not equality: new exports are additive and may land at any time,
    # while removing a name from the snapshot is a breaking change that requires
    # the documented deprecation period of at least one minor release
    # (specification 05 D3) plus a changelog entry.
    missing = sorted(set(names) - set(penampakan.__all__))
    assert not missing, f"exports removed without a deprecation period: {missing}"


# Type aliases cannot carry an own docstring; each is documented where it is
# defined in ``penampakan.models``.
_ALIAS_EXPORTS = frozenset(
    {
        "ImageSource",
        "JsonValue",
        "ObservationPayload",
        "PolicyAction",
        "VisionRequest",
    }
)


def test_every_top_level_export_documents_itself() -> None:
    # Specification 05 D3: every top-level name has a useful docstring. Only an
    # own docstring counts, because an inherited one describes a base class
    # rather than the exported name. Type aliases are the sole exemption: an
    # alias object cannot carry its own docstring, so it is documented at its
    # definition site instead, and is required here to be a genuine alias rather
    # than an undocumented class.
    undocumented = []
    aliases = []
    for name in penampakan.__all__:
        if name.startswith("__"):
            continue
        value = getattr(penampakan, name)
        own = value.__dict__.get("__doc__") if hasattr(value, "__dict__") else None
        if isinstance(own, str) and own.strip():
            continue
        if inspect.isclass(value) or inspect.isroutine(value):
            undocumented.append(name)
        else:
            aliases.append(name)

    assert not undocumented, f"top-level exports without an own docstring: {undocumented}"
    # A new export may not quietly join the exemption by being neither a class
    # nor a function.
    assert set(aliases) <= _ALIAS_EXPORTS, sorted(set(aliases) - _ALIAS_EXPORTS)


# ``.+?`` rather than ``[^:]+`` so a drive-qualified Windows path still matches.
_REVEALED = re.compile(r'^(?P<file>.+?):(?P<line>\d+): note: Revealed type is "(?P<type>.*)"$')

_OPTIONAL_CLASSES = (
    ("penampakan.llms", "OpenAITextLLM", "penampakan.llms.openai.OpenAITextLLM"),
    ("penampakan.llms", "AnthropicTextLLM", "penampakan.llms.anthropic.AnthropicTextLLM"),
    ("penampakan.llms", "LiteLLMTextLLM", "penampakan.llms.litellm.LiteLLMTextLLM"),
    ("penampakan.backends", "TesseractBackend", "penampakan.backends.tesseract.TesseractBackend"),
    (
        "penampakan.backends",
        "TransformersCaptionBackend",
        "penampakan.backends.transformers.TransformersCaptionBackend",
    ),
    (
        "penampakan.backends",
        "TransformersDetectionBackend",
        "penampakan.backends.transformers.TransformersDetectionBackend",
    ),
)


def test_optional_classes_resolve_precisely_for_a_strict_type_checker(tmp_path: Path) -> None:
    if importlib.util.find_spec("mypy") is None and shutil.which("mypy") is None:
        pytest.skip("mypy is not installed in this environment")

    probe = tmp_path / "probe.py"
    lines = ["from __future__ import annotations", ""]
    for module in ("penampakan.backends", "penampakan.llms"):
        names = sorted(name for owner, name, _ in _OPTIONAL_CLASSES if owner == module)
        lines.append(f"from {module} import {', '.join(names)}")
    lines.append("")
    lines.extend(f"reveal_type({name})" for _, name, _ in _OPTIONAL_CLASSES)
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")

    environment = dict(os.environ, MYPYPATH=str(_ROOT / "src"))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-error-summary",
            f"--cache-dir={tmp_path / 'mypy_cache'}",
            str(probe),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        cwd=tmp_path,
        env=environment,
    )
    assert "Revealed type" in completed.stdout, completed.stdout + completed.stderr

    revealed = [
        match.group("type")
        for line in completed.stdout.splitlines()
        if (match := _REVEALED.match(line)) is not None
    ]
    assert len(revealed) == len(_OPTIONAL_CLASSES), completed.stdout

    for (_, name, qualified), actual in zip(_OPTIONAL_CLASSES, revealed, strict=True):
        # A blanket ``Any`` would satisfy any assertion about the class name, so
        # compare the constructor's return type with the concrete class instead.
        returned = actual.rsplit("->", 1)[-1].strip()
        assert returned == qualified, f"{name} revealed as {actual!r}"
