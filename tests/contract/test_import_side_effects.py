"""Contract tests for the import side effects forbidden by specification 05 D4.

`import penampakan` must not import Torch, Transformers, Tesseract wrappers,
provider SDKs, LiteLLM, or OpenTelemetry, and must not read credentials, create
files, open sockets, or configure global logging.

Every case runs in a fresh subprocess so the result cannot be contaminated by
modules, environment mutations, or logging handlers that other tests already
installed in the pytest process.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Optional or heavyweight roots that a base import must never pull in. The
# provider SDKs are installed in the development environment, so this list is
# not vacuous: a stray top-level `import openai` would fail the test.
FORBIDDEN_ROOTS = (
    "torch",
    "transformers",
    "pytesseract",
    "huggingface_hub",
    "openai",
    "anthropic",
    "litellm",
    "opentelemetry",
)

LIGHTWEIGHT_MODULES = ("penampakan", "penampakan.backends", "penampakan.llms")

CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

SENTINEL_CREDENTIALS = {
    "OPENAI_API_KEY": "sentinel-openai",
    "ANTHROPIC_API_KEY": "sentinel-anthropic",
    "AWS_SECRET_ACCESS_KEY": "sentinel-aws",
    "HF_TOKEN": "sentinel-huggingface",
}


def _run(body: str, *, cwd: Path | None = None, arguments: tuple[str, ...] = ()) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        cwd=None if cwd is None else str(cwd),
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_at_least_one_forbidden_package_is_installed_so_the_probe_has_teeth() -> None:
    output = _run(
        f"""
        import importlib.util

        present = [
            name for name in {FORBIDDEN_ROOTS!r} if importlib.util.find_spec(name) is not None
        ]
        print(",".join(present))
        """
    )
    installed = [name for name in output.strip().split(",") if name]
    if not installed:
        # A base install has none of them, which is a valid environment: the
        # remaining tests still hold, they just cannot prove the probe has teeth.
        pytest.skip("no forbidden package is installed, so the probe cannot be proven non-vacuous")


@pytest.mark.parametrize("module", LIGHTWEIGHT_MODULES)
def test_importing_a_public_namespace_loads_no_forbidden_module(module: str) -> None:
    output = _run(
        f"""
        import importlib
        import sys

        before = set(sys.modules)
        importlib.import_module({module!r})
        offenders = sorted(
            name
            for name in set(sys.modules) - before
            if name.split(".")[0] in {FORBIDDEN_ROOTS!r}
        )
        print(",".join(offenders))
        """
    )
    offenders = [name for name in output.strip().split(",") if name]
    assert offenders == [], f"import {module} loaded forbidden modules: {', '.join(offenders)}"


def test_importing_the_package_writes_no_file_in_the_working_directory(tmp_path: Path) -> None:
    output = _run(
        """
        import sys

        sys.dont_write_bytecode = True
        import penampakan

        print(penampakan.__name__)
        """,
        cwd=tmp_path,
        arguments=("-B",),
    )
    assert output.strip() == "penampakan"
    assert sorted(tmp_path.iterdir()) == []


def test_importing_the_package_opens_no_network_connection() -> None:
    # Importing the stdlib `socket`/`ssl` modules is expected (asyncio pulls them
    # in); only an actual connection attempt is a contract violation.
    output = _run(
        """
        import socket


        def _forbidden(*args, **kwargs):
            raise AssertionError("import penampakan attempted a network connection")


        socket.socket.connect = _forbidden
        socket.socket.connect_ex = _forbidden
        socket.create_connection = _forbidden

        import penampakan

        print(penampakan.__name__)
        """
    )
    assert output.strip() == "penampakan"


def test_importing_the_package_reads_no_credential_environment_variable() -> None:
    output = _run(
        f"""
        import os


        class RecordingEnvironment(dict):
            def __init__(self, values):
                super().__init__(values)
                self.reads = []

            def __getitem__(self, key):
                self.reads.append(key)
                return super().__getitem__(key)

            def get(self, key, default=None):
                self.reads.append(key)
                return super().get(key, default)

            def __contains__(self, key):
                self.reads.append(key)
                return super().__contains__(key)


        environment = RecordingEnvironment({{**os.environ, **{SENTINEL_CREDENTIALS!r}}})
        os.environ = environment
        os.getenv = lambda key, default=None: environment.get(key, default)

        import penampakan  # noqa: F401

        credential_reads = sorted(
            {{
                key
                for key in environment.reads
                if any(marker in key.upper() for marker in {CREDENTIAL_MARKERS!r})
            }}
        )
        print(",".join(credential_reads))
        """
    )
    reads = [name for name in output.strip().split(",") if name]
    assert reads == [], f"import penampakan read credential variables: {', '.join(reads)}"


def test_importing_the_package_configures_no_global_logging() -> None:
    output = _run(
        """
        import json
        import logging

        calls = []
        logging.basicConfig = lambda *args, **kwargs: calls.append((args, kwargs))
        root_level_before = logging.root.level

        import penampakan  # noqa: F401

        print(
            json.dumps(
                {
                    "basic_config_calls": len(calls),
                    "root_handlers": [type(handler).__name__ for handler in logging.root.handlers],
                    "package_handlers": [
                        type(handler).__name__
                        for handler in logging.getLogger("penampakan").handlers
                    ],
                    "root_level_changed": logging.root.level != root_level_before,
                    "root_level": logging.root.level,
                }
            )
        )
        """
    )
    report = json.loads(output)
    assert report["basic_config_calls"] == 0, "import penampakan called logging.basicConfig"
    assert report["root_handlers"] == [], f"root logger gained handlers: {report['root_handlers']}"
    assert report["package_handlers"] == [], (
        f"the penampakan logger gained handlers: {report['package_handlers']}"
    )
    assert report["root_level_changed"] is False
    assert report["root_level"] == logging.WARNING
