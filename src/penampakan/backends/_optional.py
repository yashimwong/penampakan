"""Construction-time optional-dependency checks for optional vision backends.

Optional backend modules must import without their optional third-party
packages, so the packages are located rather than imported here: locating a
module answers the installation question without executing heavy package
initialization such as Torch. Construction raises the actionable configuration
error, while first use keeps its own ``BackendUnavailableError`` path because a
package can disappear or break between construction and analysis.
"""

from __future__ import annotations

import importlib.util

from penampakan.errors import ConfigurationError

__all__ = ("require_extra",)


def require_extra(extra: str, *modules: str) -> None:
    """Raise the actionable error when any module backing an extra is absent.

    ``extra`` names the installable extra (for example ``"ocr"``) and
    ``modules`` are the importable top-level module names it provides. Nothing
    is imported: a missing, broken, or unlocatable module spec is reported as
    ``ConfigurationError(code="missing_optional_dependency")``.
    """
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            # The extra name is a static library constant, never caller data.
            raise ConfigurationError(
                code="missing_optional_dependency",
                cause_summary=extra,
                extra=extra,
            )
