# Public contracts and compatibility

The public API has three tiers:

1. `penampakan` contains the clients, common settings, data contracts, errors,
   protocols, and base-install construction helpers used by the happy path.
2. Documented namespaces contain stable advanced APIs, including
   `penampakan.backends`, `penampakan.llms`, `penampakan.evaluation`,
   `penampakan.perception.cache`, and `penampakan.reasoning`.
3. A module or name beginning with `_` is private and has no compatibility
   promise.

The definitive symbol list is each module's `__all__`; signatures and model
fields are defined by code and docstrings rather than duplicated here. The
[top-level exports](../src/penampakan/__init__.py),
[domain models](../src/penampakan/models.py), and
[protocol definitions](../src/penampakan/protocols.py) are the maintained
sources. Pydantic contracts can produce exact JSON Schema at runtime, for
example `VisionAnswer.model_json_schema()`.

## Contract policy

Public Pydantic models are strict, frozen, and reject unknown fields. Wire data
uses explicit discriminators, bounded strings and collections, finite numbers,
normalized coordinates, and immutable tuples. Geometry is expressed relative
to the post-orientation asset: `(0, 0)` is the top-left and `(1, 1)` is the
bottom-right; boxes have non-empty `x_min/y_min/x_max/y_max` extents.

Serialized JSON Schema for public contract models is part of the documented
API. Adding an optional/defaulted field or enum member is a compatible feature
change; removing or renaming a field, changing its meaning, or tightening
accepted data is a breaking change. Provider-lowered schemas and private prompt
envelopes are versioned implementation contracts, not interchangeable with the
provider-neutral public schema.

Documented public tiers follow semantic versioning. A planned removal or
incompatible rename is deprecated for at least one minor release before removal.
A security or correctness emergency may skip that period only when the exception
and migration are recorded in the [changelog](../CHANGELOG.md). During the
pre-1.0 period, callers should still pin a compatible minor version for
production use.

Prompt versions are behavioral compatibility identifiers. Supported values are
discoverable through `penampakan.reasoning.supported_prompt_versions()`, and
`AgentSettings.prompt_version` rejects unsupported values during client
construction. An old supported version keeps its prompt/schema behavior through
its deprecation period. A default prompt change requires an evaluation artifact
and architecture decision record.

## Protocol boundaries

The core protocols are intentionally small:

- `TextLLM.complete(LLMRequest) -> LLMResponse`
- `VisionBackend.descriptor`, `supports`, `analyze`, and `aclose`
- `ActionPolicy.next_action(PolicyInput)`
- `Cache.get`, `set`, and `aclose`
- `TraceSink.emit` and `aclose`

Application functions can be adapted with `CallableTextLLM` and
`CallableVisionBackend`. Implementations must return the exact typed result;
core normalization treats backend drafts and model text as untrusted input.

Optional adapter classes can be imported on a base installation. Constructing
one without its declared extra raises
`ConfigurationError(code="missing_optional_dependency")`; importing the package
never reads provider credentials. Supported imports and extras are:

| API | Import path | Extra / side effect at construction |
| --- | --- | --- |
| Pillow and callable backends | `penampakan.backends` | Base; no credentials or model loading |
| Tesseract backend | `penampakan.backends` | `ocr`; Python dependency and system executable are checked on first analysis |
| Transformers backends | `penampakan.backends` | `transformers`; local model loading starts on first analysis |
| Callable text LLM | `penampakan.llms` | Base; wraps caller code |
| OpenAI / Anthropic / LiteLLM adapters | `penampakan.llms` | Corresponding provider extra; SDK client construction may resolve SDK configuration from the environment |
| Metrics | `penampakan.evaluation` | Base; experimental pure diagnostics |
| Process-local cache implementations | `penampakan.perception.cache` | Base; selected cache retains data until client close |

See [Architecture: resource ownership](architecture.md#resource-ownership) for
context-manager and close behavior.

## Evidence contract

An answered `VisionAnswer` normally requires at least one evidence reference.
Core validation proves each cited observation exists, was visible to the policy,
is not a warning, and descends from the active root asset. Evidence snapshots
carry their supporting claim and full provenance. These are structural checks;
they do not establish that an observation entails a claim or that the backend's
observation matches visual truth.
