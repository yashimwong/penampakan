"""Contract tests for a base install without any provider package.

Each case runs in a subprocess whose import machinery hides every optional
provider package, so the tests hold even when the packages are installed for the
real-SDK integration tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_BLOCKED = ("openai", "anthropic", "litellm")
_BLOCKED_BACKENDS = ("pytesseract", "torch", "transformers", "huggingface_hub")

_BLOCKER = """
import sys


class _BlockProviders:
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
sys.meta_path.insert(0, _BlockProviders())
"""


def _run(body: str, blocked: tuple[str, ...] = _BLOCKED) -> str:
    script = _BLOCKER.format(blocked=set(blocked)) + textwrap.dedent(body)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_every_provider_package_is_absent_in_the_probe_environment() -> None:
    output = _run(
        """
        import importlib

        for name in ("openai", "anthropic", "litellm"):
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            raise AssertionError(name)
        print("blocked")
        """
    )
    assert output.strip() == "blocked"


def test_adapter_modules_import_without_their_provider_package() -> None:
    output = _run(
        """
        import importlib

        for name in (
            "penampakan.llms.openai",
            "penampakan.llms.anthropic",
            "penampakan.llms.litellm",
            "penampakan.llms.schema",
        ):
            importlib.import_module(name)
        print("imported")
        """
    )
    assert output.strip() == "imported"


def test_adapter_classes_are_importable_from_the_package() -> None:
    output = _run(
        """
        from penampakan.llms import AnthropicTextLLM, LiteLLMTextLLM, OpenAITextLLM

        print(",".join(item.__name__ for item in (OpenAITextLLM, AnthropicTextLLM, LiteLLMTextLLM)))
        """
    )
    assert output.strip() == "OpenAITextLLM,AnthropicTextLLM,LiteLLMTextLLM"


def test_star_import_and_dir_work_for_the_package_and_adapters() -> None:
    output = _run(
        """
        import penampakan
        import penampakan.llms

        namespace = {}
        exec("from penampakan import *", namespace)
        exec("from penampakan.llms import *", namespace)
        assert "AsyncPenampakan" in namespace
        assert "RetryPolicy" in namespace
        assert "SchemaEnforcement" in namespace
        for name in ("OpenAITextLLM", "AnthropicTextLLM", "LiteLLMTextLLM"):
            assert name in namespace, name
            assert name in dir(penampakan.llms), name
        for name in penampakan.__all__:
            assert hasattr(penampakan, name), name
        for name in penampakan.llms.__all__:
            assert hasattr(penampakan.llms, name), name
        print("ok")
        """
    )
    assert output.strip() == "ok"


def test_documentation_and_type_introspection_work_without_provider_packages() -> None:
    output = _run(
        """
        import pydoc
        import typing

        import penampakan.llms

        for name in ("openai", "anthropic", "litellm"):
            rendered = pydoc.render_doc(f"penampakan.llms.{name}")
            assert "TextLLM" in rendered, name
        hints = typing.get_type_hints(penampakan.llms.OpenAITextLLM.complete)
        assert hints["return"].__name__ == "LLMResponse"
        print("documented")
        """
    )
    assert output.strip() == "documented"


def test_constructing_an_adapter_reports_the_missing_extra() -> None:
    # Specification 05 acceptance criterion 4: construction must fail with the
    # correct extra name. The extra is a static library constant, so naming it is
    # the one detail that makes the failure actionable; prompt, schema, and
    # credential text stay redacted.
    output = _run(
        """
        from penampakan.errors import ConfigurationError
        from penampakan.llms import AnthropicTextLLM, LiteLLMTextLLM, OpenAITextLLM

        cases = (
            (OpenAITextLLM, {"model": "gpt-4.1"}, "openai"),
            (AnthropicTextLLM, {"model": "claude-opus-5"}, "anthropic"),
            (LiteLLMTextLLM, {"model": "gpt-4o"}, "litellm"),
        )
        reported = []
        for factory, options, extra in cases:
            try:
                factory(**options)
            except ConfigurationError as error:
                assert error.code == "missing_optional_dependency", error.code
                assert error.extra == extra, error.extra
                assert f"penampakan[{extra}]" in str(error), str(error)
                assert options["model"] not in str(error)
                assert error.cause_summary == "cause details redacted"
                reported.append(f"{error.code}:{error.extra}")
            else:
                raise AssertionError(factory.__name__)
        print(",".join(reported))
        """
    )
    assert output.strip() == (
        "missing_optional_dependency:openai,"
        "missing_optional_dependency:anthropic,"
        "missing_optional_dependency:litellm"
    )


def test_optional_backend_modules_import_without_their_packages() -> None:
    output = _run(
        """
        import importlib

        for name in ("penampakan.backends.tesseract", "penampakan.backends.transformers"):
            importlib.import_module(name)
        from penampakan.backends import (
            TesseractBackend,
            TransformersCaptionBackend,
            TransformersDetectionBackend,
        )

        print(",".join(
            item.__name__
            for item in (
                TesseractBackend,
                TransformersCaptionBackend,
                TransformersDetectionBackend,
            )
        ))
        """,
        blocked=_BLOCKED_BACKENDS,
    )
    assert output.strip() == (
        "TesseractBackend,TransformersCaptionBackend,TransformersDetectionBackend"
    )


def test_constructing_an_optional_backend_reports_the_missing_extra() -> None:
    output = _run(
        """
        import sys

        from penampakan.backends import (
            TesseractBackend,
            TransformersCaptionBackend,
            TransformersDetectionBackend,
        )
        from penampakan.errors import ConfigurationError

        cases = (
            (TesseractBackend, "ocr"),
            (TransformersCaptionBackend, "transformers"),
            (TransformersDetectionBackend, "transformers"),
        )
        reported = []
        for factory, extra in cases:
            try:
                factory()
            except ConfigurationError as error:
                assert error.code == "missing_optional_dependency", error.code
                assert error.extra == extra, error.extra
                assert f"penampakan[{extra}]" in str(error), str(error)
                reported.append(error.extra)
            else:
                raise AssertionError(factory.__name__)
        # A failed construction must not have loaded any heavyweight package.
        assert {"torch", "transformers", "huggingface_hub"}.isdisjoint(sys.modules)
        print(",".join(reported))
        """,
        blocked=_BLOCKED_BACKENDS,
    )
    assert output.strip() == "ocr,transformers,transformers"


def test_the_schema_compiler_works_without_any_provider_package() -> None:
    output = _run(
        """
        from penampakan.llms.schema import SchemaTarget, compile_action_schema
        from penampakan.models import Capability
        from penampakan.perception.registry import ToolRegistry
        from penampakan.reasoning.prompts import build_action_schema
        from penampakan.tools.builtin import register_transform_tools
        from penampakan.tools.vision import register_vision_tools

        registry = ToolRegistry()
        register_vision_tools(registry, set(Capability))
        register_transform_tools(registry)
        schema = build_action_schema(registry.specs)
        fingerprints = {
            compile_action_schema(schema, target=target).fingerprint_sha256
            for target in SchemaTarget
        }
        print(len(fingerprints))
        """
    )
    assert output.strip() == "3"


def test_strict_type_checking_tolerates_absent_provider_packages() -> None:
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    configuration = tomllib.loads(project.read_text(encoding="utf-8"))
    mypy = configuration["tool"]["mypy"]

    assert mypy["strict"] is True
    tolerated = {
        module
        for override in mypy["overrides"]
        if override.get("ignore_missing_imports") is True
        for module in override["module"]
    }
    # An unresolved provider import must not fail strict type checking on a base
    # install; an installed SDK is still checked against its real types.
    for package in _BLOCKED:
        assert f"{package}.*" in tolerated, package
