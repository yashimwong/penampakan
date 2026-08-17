from __future__ import annotations

import os
from collections import defaultdict

import pytest

_TRUTHY = {"1", "all", "true", "yes"}


def _integration_categories(item: pytest.Item) -> set[str]:
    marker = item.get_closest_marker("integration")
    if marker is None:
        return set()
    category_marker = item.get_closest_marker("integration_category")
    category_values = category_marker.args if category_marker is not None else marker.args
    categories = {str(value).strip() for value in category_values if str(value).strip()}
    if categories:
        return categories
    inherited = {name for name in ("ocr", "models") if item.get_closest_marker(name)}
    return inherited or {"integration"}


def _required_categories(value: str, available: set[str]) -> set[str]:
    normalized = value.strip().lower()
    if not normalized:
        return set()
    if normalized in _TRUTHY:
        return available or {"integration"}
    return {category.strip() for category in value.split(",") if category.strip()}


def pytest_configure(config: pytest.Config) -> None:
    config._penampakan_integration_categories = {}
    config._penampakan_integration_reports = defaultdict(dict)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    categories: dict[str, set[str]] = config._penampakan_integration_categories
    for item in items:
        item_categories = _integration_categories(item)
        if item_categories:
            categories[item.nodeid] = item_categories


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> pytest.TestReport:
    report = yield
    reports: dict[str, dict[str, pytest.TestReport]] = item.config._penampakan_integration_reports
    if report.nodeid in item.config._penampakan_integration_categories:
        reports[report.nodeid][report.when] = report
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    categories: dict[str, set[str]] = config._penampakan_integration_categories
    required = _required_categories(
        os.getenv("PENAMPAKAN_REQUIRE_INTEGRATION", ""),
        set().union(*categories.values()) if categories else set(),
    )
    if not required:
        return
    completed: set[str] = set()
    reports: dict[str, dict[str, pytest.TestReport]] = config._penampakan_integration_reports
    for nodeid, node_reports in reports.items():
        setup = node_reports.get("setup")
        call = node_reports.get("call")
        if (setup is not None and setup.failed) or (call is not None and not call.skipped):
            completed.update(categories[nodeid])
    missing = sorted(required - completed)
    if not missing:
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    message = "required integration categories had no non-skipped outcome: " + ", ".join(missing)
    if reporter is not None:
        reporter.write_sep("ERROR", message, red=True)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
