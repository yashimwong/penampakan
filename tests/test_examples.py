from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = tuple(sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")))
OFFLINE_OUTPUTS = {
    "01_inspect_without_llm.py": ("image=64x40", "observations=metadata,colors"),
    "04_custom_backend.py": ("status=answered", "evidence=obs_000001"),
    "06_tracing.py": ("events=", "redacted={}"),
    "07_budgets_and_abstention.py": (
        "status=insufficient_evidence",
        "stop_reason=insufficient_evidence",
        "llm_calls=1/2",
    ),
    "08_multi_image.py": ("SKIPPED: multi-image sessions require specification 10",),
}
# A subprocess example runs with a deliberately minimal environment so it cannot
# read a credential, but an interpreter still needs a few platform variables
# before it will start at all. Windows in particular resolves system libraries
# through SYSTEMROOT.
_PLATFORM_REQUIRED_VARIABLES = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "LD_LIBRARY_PATH",
)


def _isolated_environment(**overrides: str) -> dict[str, str]:
    """Return a credential-free subprocess environment that can start Python."""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
    }
    for name in _PLATFORM_REQUIRED_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    environment.update(overrides)
    assert not any(
        marker in name for name in environment for marker in ("KEY", "TOKEN", "CREDENTIAL")
    )
    return environment


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_import_is_safe(
    path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for name in tuple(os.environ):
        if "KEY" in name or "TOKEN" in name or "CREDENTIAL" in name:
            monkeypatch.delenv(name, raising=False)
    before = tuple(tmp_path.iterdir())
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None))
    assert capsys.readouterr() == ("", "")
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize("name,expected", OFFLINE_OUTPUTS.items())
def test_offline_example_subprocess(name: str, expected: tuple[str, ...], tmp_path: Path) -> None:
    environment = _isolated_environment(PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert all(fragment in completed.stdout for fragment in expected)
    assert tuple(tmp_path.iterdir()) == ()


def test_all_expected_examples_exist() -> None:
    assert tuple(path.name for path in EXAMPLES) == tuple(
        f"{index:02d}_{suffix}.py"
        for index, suffix in enumerate(
            (
                "inspect_without_llm",
                "ask_with_openai",
                "ask_with_local_models",
                "custom_backend",
                "reusable_session",
                "tracing",
                "budgets_and_abstention",
                "multi_image",
            ),
            start=1,
        )
    )
    assert all(len(path.read_text(encoding="utf-8").splitlines()) <= 80 for path in EXAMPLES)


def test_readme_quickstart_matches_example_and_executes(tmp_path: Path) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    marked = readme.split("<!-- quickstart:start -->", 1)[1].split("<!-- quickstart:end -->", 1)[0]
    snippet = marked.strip().removeprefix("```python\n").removesuffix("```").rstrip()
    source = (ROOT / "examples" / "01_inspect_without_llm.py").read_text(encoding="utf-8")
    assert snippet in source
    completed = subprocess.run(
        [sys.executable, "-c", f"{snippet}\n\nmain()\n"],
        cwd=tmp_path,
        env=_isolated_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "image=64x40\nobservations=metadata,colors\n"
    assert tuple(tmp_path.iterdir()) == ()
