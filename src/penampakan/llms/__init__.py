"""Text language model adapters.

Every provider adapter class imports on a base install, because no adapter
module imports its optional provider SDK at module import time. Instantiating an
adapter without its extra raises
``ConfigurationError(code="missing_optional_dependency")``.
"""

from penampakan.llms.anthropic import AnthropicTextLLM
from penampakan.llms.callable import CallableTextLLM
from penampakan.llms.litellm import LiteLLMTextLLM
from penampakan.llms.openai import OpenAITextLLM
from penampakan.llms.schema import (
    SCHEMA_COMPILER_VERSION,
    CompiledSchema,
    SchemaTarget,
    compile_action_schema,
    unwrap_action_envelope,
)

__all__ = [
    "SCHEMA_COMPILER_VERSION",
    "AnthropicTextLLM",
    "CallableTextLLM",
    "CompiledSchema",
    "LiteLLMTextLLM",
    "OpenAITextLLM",
    "SchemaTarget",
    "compile_action_schema",
    "unwrap_action_envelope",
]
