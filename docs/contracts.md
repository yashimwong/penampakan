# Public contracts and compatibility

The public API has three tiers:

1. `penampakan` contains the clients, common settings, data contracts, errors,
   protocols, and base-install construction helpers used by the happy path.
2. Documented namespaces contain stable advanced APIs: `penampakan.backends`,
   `penampakan.llms`, `penampakan.evaluation`, `penampakan.image`,
   `penampakan.perception.cache`, `penampakan.perception.sqlite_cache`,
   `penampakan.reasoning`, `penampakan.reasoning.policy`, and
   `penampakan.tracing`. Adding a backend,
   provider, cache, evaluator, parser, prompt helper, or trace sink extends the
   namespace that owns it rather than the top level, so tier 1 stays small
   enough to remain discoverable.
3. A module or name beginning with `_` is private and has no compatibility
   promise. So is any module not named above: `penampakan.perception` and
   `penampakan.tools` are package containers that deliberately export nothing,
   and their unlisted submodules — routing, normalization, storage — are
   implementation detail. `penampakan.perception.sqlite_cache` is the one
   exception, listed above because an operator administering a durable cache has
   to import the implementation from somewhere.

The definitive symbol list is each module's `__all__`; every tier-1 and tier-2
module defines one explicitly, and an absent or empty `__all__` is an intentional
statement that the module exports nothing. Signatures and model fields are
defined by code and docstrings rather than duplicated here. The
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
- `ManagedCache.stats`, `clear`, and `prune`, in addition to the `Cache` surface
- `TraceSink.emit` and `aclose`

A session only ever uses `Cache`, where a failure degrades to a miss or a no-op.
`ManagedCache` is the administrative surface: it raises typed errors instead,
because an operator who called `stats`, `clear`, or `prune` is misled by silent
failure. `Cache.set` validates that the supplied `size` equals the length of the
value, and an implementation persists the size it verified rather than the number
it was given.

Application functions can be adapted with `CallableTextLLM` and
`CallableVisionBackend`. Implementations must return the exact typed result;
core normalization treats backend drafts and model text as untrusted input.

Every optional adapter class can be imported on a base installation, because no
adapter module imports its optional third-party package at module import time.
Constructing one without its declared extra raises
`ConfigurationError(code="missing_optional_dependency")`, which names the extra
on `ConfigurationError.extra` and in its public message as
`Install penampakan[<extra>].`. The extra is a static library constant, so it is
the one detail a configuration failure reports verbatim; prompt, schema, and
credential text stay redacted. Importing the package never reads credentials,
opens a connection, writes a file, loads model weights, or configures global
logging or telemetry.

| API | Import path | Extra | Construction side effects | Ownership |
| --- | --- | --- | --- | --- |
| Pillow and callable backends | `penampakan.backends` | Base | None | Caller-owned; `aclose` |
| Tesseract backend | `penampakan.backends` | `ocr` | The `ocr` extra is verified; the system executable and language data are diagnosed on first analysis | Caller-owned; `aclose` |
| Transformers backends | `penampakan.backends` | `transformers` | The `transformers` extra is verified and the model revision is resolved; weights load on first analysis | Caller-owned; `aclose` |
| Callable text LLM | `penampakan.llms` | Base | Wraps caller code | Caller-owned; `aclose` and async context manager |
| OpenAI / Anthropic / LiteLLM adapters | `penampakan.llms` | Matching provider extra | SDK client construction may resolve SDK configuration from the environment | Caller-owned unless `owns_llm=True`; async context manager |
| Action policy | `penampakan.reasoning.policy` | Base | None; validates `prompt_version` against `supported_prompt_versions()` | Caller-owned unless `owns_policy=True`; closes its LLM only with `owns_llm=True` |
| Prompt-version discovery | `penampakan.reasoning` | Base | None; pure functions and constants | Not applicable |
| Metrics | `penampakan.evaluation` | Base | None; experimental pure diagnostics | Not applicable |
| Image loading, geometry, and Set-of-Mark rendering | `penampakan.image` | Base | None; pure functions over caller data; `mark_regions` uses built-in vector digits and no font asset | Returned assets are plain values; callers close the pending marked asset |
| Trace building and redaction | `penampakan.tracing` | Base | None; a builder holds only run-local state | Owned by the run that created it; caller-supplied `TraceSink`s stay caller-owned |
| In-memory and JSONL trace sinks | `penampakan.trace_sinks` | Base | JSONL starts its writer lazily on first emit | Caller-owned unless `owns_trace_sinks=True`; close drains accepted JSONL events |
| OpenTelemetry trace sink | `penampakan.trace_sinks` | `opentelemetry` | Validates the injected provider; never changes global OTel state | Caller-owned unless `owns_trace_sinks=True` |
| Process-local cache implementations | `penampakan.perception.cache` | Base | None | Owned by the client that receives it; retains data until client close |
| Durable SQLite cache | `penampakan.perception.sqlite_cache` | Base | Starts one worker thread, creates a private parent directory and database file, and opens the database; a path or data failure disables the instance instead of raising | Owned by the client that receives it; `aclose` drains queued work and stops the worker, but retained data outlives the process |

A tier-2 name is subject to the same semantic-versioning and deprecation rules
as a tier-1 name. See
[Architecture: resource ownership](architecture.md#resource-ownership) for the
full context-manager and close ordering rules; no shared adapter or external SDK
client is ever closed implicitly.

## Evidence contract

An answered `VisionAnswer` normally requires at least one evidence reference.
Core validation proves each cited observation exists, was visible to the policy,
is not a warning, and descends from the active root asset. Evidence snapshots
carry their supporting claim and full provenance. These are structural checks;
they do not establish that an observation entails a claim or that the backend's
observation matches visual truth.

`MarkRef` and `MarkPayload` are top-level public contracts. A `MarkRef` contains a
contiguous one-based index, the original observation ID, its normalized `Box`,
and an optional source label. A `MarkPayload` spans the derived asset, contains
one to 99 unique references, and deliberately has no `Observation.region` of its
own. It is a machine-readable transform mapping, not perception evidence, and
core evidence validation rejects it just as it rejects a `TransformPayload`.

`CaptionRequest.mark_indices` selects up to 99 unique rendered indices for a
backend whose caption capability advertises `caption.mark_references`. Such a
request must return one structured `MarkDescriptionPayload`, containing unique
`MarkDescriptionRef(index, description)` values, rather than an ordinary caption
or free text that merely mentions a number. Mark descriptions are untrusted
backend observations; unlike the transform-only `MarkPayload`, they can be cited
alongside the original localized observations when they actually support a
claim.

`penampakan.image.mark_regions(image, regions, ...)` is the separate
programmatic raw-box API. It returns a pending `set_of_mark` asset whose
descriptor identifies `source="caller_supplied"`; it does not manufacture
observation IDs or a `MarkPayload`. Region order cannot be used to select badge
indices: normalized boxes are sorted spatially, exact and configurable
near-duplicates are removed, and indices are assigned after deterministic
placement. `include_labels=False` is the default. This function is available
without an LLM or optional model package, and the returned pending asset owns its
rendered Pillow image until the caller closes it.
