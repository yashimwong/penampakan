"""Bounded action policy and evidence reasoning."""

from penampakan.reasoning.prompts import PROMPT_VERSION, SUPPORTED_PROMPT_VERSIONS


def supported_prompt_versions() -> tuple[str, ...]:
    """Return behaviorally compatible prompt versions accepted by this release."""
    return SUPPORTED_PROMPT_VERSIONS


__all__ = ("PROMPT_VERSION", "supported_prompt_versions")
