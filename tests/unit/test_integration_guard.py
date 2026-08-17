from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import cast

import pytest

import tests.conftest as integration_plugin


class _Reporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def write_sep(self, separator: str, message: str, **kwargs: object) -> None:
        self.messages.append(message)


def _session(*, skipped: bool) -> tuple[pytest.Session, _Reporter]:
    reporter = _Reporter()
    config = SimpleNamespace(
        _penampakan_integration_categories={"test_sample.py::test_case": {"ocr"}},
        _penampakan_integration_reports=defaultdict(
            dict,
            {
                "test_sample.py::test_case": {
                    "setup": SimpleNamespace(failed=False),
                    "call": SimpleNamespace(skipped=skipped),
                }
            },
        ),
        pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter),
    )
    return cast(pytest.Session, SimpleNamespace(config=config, exitstatus=0)), reporter


def test_required_integration_guard_rejects_all_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PENAMPAKAN_REQUIRE_INTEGRATION", "ocr")
    session, reporter = _session(skipped=True)

    integration_plugin.pytest_sessionfinish(session, 0)

    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert reporter.messages == ["required integration categories had no non-skipped outcome: ocr"]


def test_required_integration_guard_accepts_call_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PENAMPAKAN_REQUIRE_INTEGRATION", "ocr")
    session, reporter = _session(skipped=False)

    integration_plugin.pytest_sessionfinish(session, 0)

    assert session.exitstatus == 0
    assert reporter.messages == []
