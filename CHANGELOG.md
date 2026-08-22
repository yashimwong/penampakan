# Changelog

## Penampakan 0.6.1
Released 2026-08-22

- Corrected the release contents: the `0.6.0` package was tagged before the
  documented Set-of-Mark implementation commit. This release includes the
  public mark contracts, deterministic mark rendering, and gated agent workflow
  described in the `0.6.0` notes below.
- Mark-aware tool registration and request routing now require a concrete model
  identity and pinned model revision in addition to the advertised
  `caption.mark_references` feature, so an unpinned backend cannot be selected
  for structured mark descriptions.
- Expanded deterministic mark-rendering coverage across dark, light, textured,
  edge, corner, tiny, overlapping, crowded, and 99-region inputs, including
  environments without an available font.

## Penampakan 0.6.0
Released 2026-08-20

- Trace events now use schema v2 with opaque invocation and parent IDs,
  correlated start/finish pairs for policy, tool, backend, and verification
  operations, and self-contained terminal summary counters. Missing versions
  remain legacy v1 for backward parsing only; their formerly valid event names,
  including dotted names, remain accepted while v2 enforces lower snake case.
- Added bounded `InMemoryTraceSink`, single-writer rotating `JsonlTraceSink`, and
  optional injected-provider `OpenTelemetryTraceSink`. JSONL defaults to
  non-blocking overflow and explicitly supports concurrent sessions only within
  one process. OpenTelemetry CI versions are pinned in the optional requirements.
- Trace finalization now finishes every still-active invocation before emitting
  the terminal event, including cancellation while a shielded single-flight
  backend population unwinds, so every start retains exactly one matching finish.
- OpenTelemetry retains a bounded set of completed invocation span contexts, so
  a tool selected by a completed policy call keeps the policy span as its causal
  parent instead of being diagnosed and reparented to the run.
- JSONL's default symlink protection now rejects links in every component of the
  configured path, not only the final filename. `allow_symlink=True` remains the
  explicit opt-in for linked destinations.
- Caller-supplied trace sinks are now caller-owned by default. Set
  `owns_trace_sinks=True` to drain and close them with the client.
- Added the public `MarkRef`, `MarkPayload`, `MarkDescriptionRef`, and
  `MarkDescriptionPayload` contracts, `CaptionRequest.mark_indices`, and the
  `set_of_mark` transform descriptor. Mark mappings preserve source observation
  IDs and regions as transform lineage, but are explicitly rejected as answer
  evidence; a rendered mapping does not prove a label or backend description.
  A mark-aware caption request instead requires bounded structured index-to-text
  observations, not free text that happens to contain a number.
- Added the base-install `penampakan.image.mark_regions` programmatic transform
  for one to 99 caller-supplied normalized boxes. It provides deterministic
  spatial numbering, exact/configurable near-duplicate removal, bounded
  collision placement, alpha preservation, optional sanitized labels, and
  font-independent vector digits covered by the project's MIT license. Its
  descriptor records `source="caller_supplied"`, and it creates no observation
  or `MarkPayload`.
- Added a gated observation-backed agent workflow: `mark_regions` accepts only
  visible detection/segmentation observation IDs, emits one transform plus one
  `MarkPayload`, and `describe_marks` requires structured mark-indexed backend
  output. The tools remain absent without localized sources and an exact backend
  advertising `caption.mark_references`. `agent-v2` adds mark-specific evidence
  guidance only while the tool is exposed; `agent-v1` and the default prompt/tool
  behavior remain unchanged. No current first-party visual backend advertises the
  feature, and the real-weight contract and paired SoM-versus-control quality gate
  remain open.

## Penampakan 0.5.0
Released 2026-08-18

Breaking: `CacheSettings.enabled` is replaced by `CacheSettings.mode`, one of
`"off"`, `"memory"`, or `"sqlite"`. `CacheSettings(enabled=True)` becomes
`CacheSettings(mode="memory")`. A cache stores derived descriptions of user
images, so retention is now selected explicitly rather than toggled, and it
remains off by default. `Cache.set` also rejects an accounted `size` that
disagrees with `len(value)`, and a value larger than the whole byte budget is
now a true no-op that leaves any value already stored under that key intact
instead of discarding it.

- Added `SQLiteCache`, a retention-aware durable perception cache, selected with
  `CacheSettings(mode="sqlite", path=...)`. One dedicated worker thread creates,
  owns, and closes the SQLite connection and processes an ordered queue, so the
  event loop never touches the connection and no operation can run on a thread
  the connection does not belong to. The worker enables WAL and a bounded busy
  timeout, uses short explicit transactions and idempotent upserts, retries lock
  contention with bounded jitter only inside the operation deadline, and
  converts exhausted contention into a degraded miss rather than a run failure.
- The durable cache disables itself instead of failing client construction when
  its database cannot be used, and reports why on `SQLiteCache.status`: a
  runtime older than SQLite 3.37 cannot create `STRICT` tables
  (`sqlite_version_unsupported`), a database written by a newer schema is opened
  for no data so an older binary cannot drop future entries
  (`cache_schema_too_new`), and an unusable path, a rejected symlink, or a
  failed quarantine each disable the instance rather than overwrite anything.
  A corrupt or non-database file is quarantined under a unique
  `.corrupt-<timestamp>-<nonce>` name before a fresh database is created.
- Durable cache entries expire on an absolute deadline measured from their
  creation, are evicted by an approximate least-recently-used order that batches
  access timestamps so a reader does not become a writer, and are bounded by
  high and low watermarks that a single accepted write may exceed only once.
  Retained byte totals are derived transactionally from verified value sizes.
- Penampakan creates a cache directory with mode `0700` and its database, WAL,
  and shared-memory files with mode `0600` where the platform supports POSIX
  permissions, warns when existing permissions are broader, and refuses a
  symlinked cache path unless `allow_symlink=True`. Cache content is not
  encrypted, and `clear()` removes entries logically rather than securely.
- Added the `ManagedCache` protocol and the `CacheStats` model for cache
  administration. `stats`, `clear`, and `prune` raise typed errors rather than
  failing silently, because an operator who calls them is owed an answer; the
  session-facing `Cache` surface still degrades to a miss or a no-op.
  `NullCache` and `MemoryLRUCache` implement it too.
- A cache failure during perception now produces exactly one redacted
  `cache_operation_failed` warning per call, plus a `cache_operation_failed`
  trace event per failed operation and a new `TraceSummary.cache_failures`
  counter. Only the error type and a library error code are reported; no cache
  key, path, or exception text reaches a warning or a trace.
- A perception result served by a shared in-flight population is no longer
  attributed to the backend that was merely preferred. The population now
  carries the cache key its serving descriptor reproduces, so a waiter that
  finds a different key re-perceives instead of recording a cache hit against a
  backend that never ran, while a waiter in any session whose key does match
  still shares the result.
- Documented what a cache stores, its content-addressed key dimensions, TTL and
  approximate-recency semantics, its filesystem artifacts and permissions, its
  lock-contention limits, and that image-derived text may be as sensitive as the
  image itself. Enabling trace content never enables a cache, and enabling a
  cache never weakens trace redaction.
- Exported `CacheStats` and `ManagedCache` from the top level and documented
  `penampakan.perception.sqlite_cache` as an advanced namespace. CI now records
  the SQLite runtime version it ran against.

## Penampakan 0.4.0
Released 2026-08-18

Breaking: an unpinned `TesseractBackend` now reports an extra trailing
`unpinned_engine_version` warning on every OCR result, and a caller-supplied
`Cache` that does not declare `durable = False` is treated as durable, so it is
bypassed for a backend whose model weight revision is unresolved. Declare
`durable = False` on a process-local cache to keep the previous behavior.

- Client shutdown no longer loses the exception it retained when the internal
  close task is itself cancelled, as `asyncio.run()` shutdown and a `TaskGroup`
  abort both do. A retained cancellation is withdrawn from the close task so it
  completes normally, and `aclose` re-raises the retained primary instead of the
  task's own `CancelledError`. Repeated closes observe the same primary.
- Shutdown now marks the client closed before releasing backend ownership, and
  records a redacted `backend_ownership` close warning if that release fails, so
  a failing release can no longer leave `closed` unset or shadow the primary
  exception.
- Cancelling a client close while owned sessions are still closing no longer
  discards the failures the finished sessions already reported. The session
  gather runs in a shielded task, every session is still attempted, and a base
  exception reported by a session takes precedence over the cancellation.
- The whole client constructor after the backend-ownership claim is now guarded,
  so a failure while constructing the built-in Pillow backend or resolving the
  authoritative metadata preference releases the claim instead of leaving a
  caller backend registered to a client that never finished constructing.
- `functools.partial` wrapping an object whose `__call__` is asynchronous is now
  detected as an async callable. It previously consumed a worker thread merely to
  construct its coroutine, because `inspect.iscoroutinefunction` unwraps a partial
  only far enough to read a function's code flags.
- Cancelling an application callable that runs in a worker thread and returns an
  awaitable now closes that awaitable instead of abandoning it unawaited.
- Every result served by a backend that declares a model identity without a
  resolved weight revision now carries `unresolved_model_revision`, whichever
  backend served it. The signal was previously emitted only by the Transformers
  adapter, so a callable or third-party backend in that state was silently
  excluded from durable caches without saying so.
- `Cache.durable` is now fail-closed: only an explicit `durable = False` opts a
  cache out. A caller-supplied durable cache that omitted the declaration was
  previously treated as ephemeral and used for backends whose weight identity is
  unresolved, contradicting the retention rule it was meant to enforce. The
  shipped `NullCache` and `MemoryLRUCache` declare `durable = False` and are
  unaffected.
- `TesseractBackend` accepts an optional `engine_version` that pins the concrete
  Tesseract engine build into `BackendDescriptor.version`, verified against the
  running binary on first analysis and failing as
  `BackendUnavailableError(code="tesseract_engine_version_mismatch")` on a
  mismatch, so two engine builds never share a durable perception cache key. An
  unpinned backend keeps its previous descriptor and now reports the probed
  engine version on every result as `WarningInfo(code="unpinned_engine_version")`.
  ADR 0003 records why the version is pinned by the caller rather than probed.

## Penampakan 0.3.0
Released 2026-08-18

Breaking: constructing `TesseractBackend`, `TransformersCaptionBackend`, or
`TransformersDetectionBackend` without its extra now fails immediately instead
of succeeding and deferring a `BackendUnavailableError` to first analysis. Code
that constructed an optional backend to probe availability must catch
`ConfigurationError` at the construction site. `ConfigurationError` messages for
a missing dependency also changed, and `JsonActionPolicy.prompt_version` now
reports the version it was configured with rather than the module default.

- Added maintained product, architecture, contract, runtime, and quality guides
  grounded in the shipped public API.
- Exposed `CallableTextLLM`, `CallableVisionBackend`, and `PillowBackend` on the
  base-install top level, and added
  `penampakan.reasoning.supported_prompt_versions()` for behavioral-version
  discovery.
- Added import-safe runnable examples for inspection, hosted and local answering,
  custom adapters, reusable sessions, tracing, bounded abstention, and the
  current single-image limitation.
- Added documentation/example validation, local Markdown link checks, package
  metadata and distribution-content release gates.
- Recorded the existing metadata orchestration-overhead result in an immutable
  manifest and clarified that smoke/contract checks are not accuracy evidence.
- Documented the tiered public API surface. `penampakan.image`,
  `penampakan.reasoning.policy`, and `penampakan.tracing` join the listed stable
  advanced namespaces, each namespace now records its import path, extra,
  construction side effects, and ownership, and `penampakan.perception` and
  `penampakan.tools` declare an intentionally empty `__all__` so their
  submodules are unambiguously implementation detail.
- Constructing `TesseractBackend`, `TransformersCaptionBackend`, or
  `TransformersDetectionBackend` without its extra now raises
  `ConfigurationError(code="missing_optional_dependency")` immediately, matching
  the optional provider adapters, instead of deferring a
  `BackendUnavailableError` to first analysis. The check locates the packages
  without importing them, so no weights load and construction stays cheap; the
  deferred unavailability path is retained for a package that disappears or
  breaks after construction.
- `ConfigurationError` now reports the installable extra on a new `extra`
  attribute and in its public message as `Install penampakan[<extra>].`. The
  extra is a static library constant validated against the same conservative
  token shape as other reported identifiers, so prompt, schema, and credential
  text remain redacted. A missing-dependency failure previously named no extra
  at all, because the free-form cause summary is redacted by design.
- `AgentSettings.prompt_version` now defaults to the canonical `PROMPT_VERSION`
  rather than a duplicated literal, and the client and `JsonActionPolicy`
  validate against `supported_prompt_versions()` membership instead of a single
  constant. `JsonActionPolicy.prompt_version` reports the version it was
  configured with rather than the module default.
- Added public API surface contract tests: every README import resolves from the
  path shown, the base-install top-level and advanced namespace imports hold with
  every optional package hidden, `dir()` and star imports are checked by required
  inclusion rather than exact equality, a checked-in export snapshot gates
  removals against the deprecation policy, and optional classes resolve
  precisely under a real strict type-checker run.
- Added import side-effect and import-performance regression gates. A base import
  is asserted to load no Torch, Transformers, Tesseract, provider SDK, or
  OpenTelemetry module and to perform no filesystem, network, credential, or
  global logging activity. Import cost is compared against a checked-in envelope
  calibrated by a control measurement taken in the same job, plus an absolute
  emergency ceiling, rather than a fixed wall-clock assertion.

## Penampakan 0.2.1
Released 2026-08-18

Real-dependency integration coverage now exercises pinned Tesseract and
Transformers environments, deterministic processor geometry, cleanup, EXIF
orientation, Arabic tessdata, and immutable model provenance. Real-weight
goldens use the public-domain NASA astronaut fixture with recorded caption and
detection revisions, confidence and IoU thresholds, attribution, and SHA-256.
The scheduled integration workflow fails when any required category only skips.

## Penampakan 0.2.0
Released 2026-08-16

Metadata inspection now reuses the loader's canonical image, uses balanced PNG
compression, and reports reusable-session latency plus executed normalization
and safety checks in the benchmark.

Client shutdown now attempts every owned resource in a documented dependency
order even when one of them fails. Ordinary close failures are recorded as
redacted warnings on the new AsyncPenampakan.close_warnings property, while the
first cancellation or other base exception is retained, propagated once the
remaining cleanup has been attempted, and never displaced by a later ordinary
failure; ADR 0002 records why the public close re-raises it. Backend ownership
moved from a process-global set of raw addresses to a weak identity registry, so
address reuse, garbage collection, unusual hashing or equality, and stale
weak-reference callbacks can no longer reject or steal an unrelated backend, and
a backend that cannot be weakly referenced is accepted instead of rejected.

Application callables supplied to the callable vision backend and the callable
text language model are now inspected before dispatch, so an asynchronous
callable, including a functools.partial or an object with an asynchronous
__call__, is awaited on the event loop instead of consuming a worker thread
merely to construct its coroutine.

Model-backed adapters now resolve the exact loaded-weight identity from an
explicit commit revision or the local Hugging Face cache snapshot, and report an
unresolved_model_revision warning on every result when they cannot. An
unresolved backend is excluded from durable caches through the new
BackendDescriptor.durable_cache_eligible property while in-process
deduplication and ephemeral memory caching remain available. The Tesseract
backend no longer claims a model identity and instead reports its language and
configuration selection in its backend version.

A new lineage benchmark drives a saturated reasoning session at the configured
maximum of sixteen derived assets and three derivation levels, then attributes
session wall time to the asset-lineage and context scans. Those scans cost
0.046 ms of a 71 ms session, or 0.065%, so the asset-lineage optimization is
deferred with a documented budget of 1.0 ms per session and 0.15 ms per
reasoning step, recorded in ADR 0001. The store-owned root lookup that
multi-image sessions require for correctness is unaffected.

Provider adapters now compile the provider-neutral action schema into a named,
versioned provider subset before any request is sent. The new
penampakan.llms.schema module lowers the root discriminated union into the
documented nested form only after proving the branches are mutually
distinguishable by a required const discriminator, closes every reachable
object, converts previously optional properties into required nullable unions
for the OpenAI target, resolves local references, enforces the documented depth,
property, enum, and size limits, verifies a round trip against the original
schema, and canonicalizes and fingerprints the result. One explicit keyword
table records every keyword that is preserved, dropped, or rejected per target,
and an unrecognized keyword fails compilation rather than being weakened
silently. Values a target cannot express are still enforced locally, because
every adapter post-validates the parsed action against the original schema even
after provider strict enforcement.

OpenAITextLLM, AnthropicTextLLM, and LiteLLMTextLLM ship as optional-extra
adapters. Each class module imports on a base install and raises
ConfigurationError(code="missing_optional_dependency") only when constructed
without its extra. OpenAITextLLM uses the OpenAI Responses API with strict
json_schema output and one centrally tested capability table for instruction
role and sampling support. AnthropicTextLLM uses native structured JSON output
when the configured model supports it and one forced strict client tool
otherwise, sends the system instruction through Anthropic's top-level system
field, and reports which strict path ran through non-sensitive response
metadata. LiteLLMTextLLM queries LiteLLM's capability metadata before a request
instead of probing by provoking errors; without strict schema support it raises
ConfigurationError(code="strict_schema_unavailable") unless allow_json_only is
explicitly set, in which case it uses JSON mode, post-validates the original
schema, and reports schema_enforcement=json_only.

LLMResponse gained provider, request_id, backend_fingerprint, attempts, and
schema_enforcement, so provider identity, retry counts, and degraded enforcement
are visible in typed state instead of logs. The new RetryPolicy bounds provider
attempts with capped exponential backoff and full jitter, and one shared retry
implementation retries only connection failures, timeouts, 429, and 5xx. A
monotonic deadline from LLMRequest.timeout_s covers every attempt, SDK call,
response read, and backoff, and a retry never begins when its minimum work would
cross the deadline. Adapters that construct their own SDK client disable that
client's native retries so provider attempts cannot multiply. LLMError now
carries a safe attempt count and the last provider status and error code without
any response content.

Ownership is explicit end to end. An adapter owns its SDK client only when it
constructed it, unless owns_client says otherwise. JsonActionPolicy gained
owns_llm and closes only an owned language model, and AsyncPenampakan and
Penampakan gained owns_policy and owns_llm. Caller-supplied resources default to
caller-owned, the convenience construction path owns only the policy it builds,
and an owned policy closes after the sessions that borrow it and before the
cache, cascading to a language model it owns. Provider adapters and the policy
are idempotent asynchronous context managers.

A run that used degraded JSON-only enforcement now carries exactly one
WarningInfo(code="degraded_schema_enforcement"). The warning travels through
typed policy and response state rather than a log containing request content.

JsonActionPolicy and build_policy_request accept an explicit temperature so a
caller can target a model whose API only accepts its own sampling default; the
deterministic default is unchanged.

Every policy call now reports its provider attempt count, token usage, and schema
enforcement into the run trace, so the trace summary's token counters are
populated and a retried provider call is visible in budgets and cost reports
while still consuming one orchestrator language-model reservation. All three
adapters classify provider failures identically: connection failures, timeouts,
408, 429, and 5xx are retried, and every other 4xx is terminal.

Provider contract tests now run only in the provider-SDK CI job. Provider-raised
timeouts remain eligible for retries under a bounded total deadline, invalid
responses retain their attempt and token metrics when repaired, fractional
multipleOf constraints use decimal arithmetic, OpenAI retry policies are
validated during construction, and strict schema compilation preserves root
descriptions.

## Penampakan 0.1.0
Released 2026-08-10

Initial release of bounded visual tool orchestration for text-only language
models. It includes strict public contracts, safe image normalization,
deterministic perception routing, evidence-grounded question answering,
reusable asynchronous and synchronous sessions, optional local model adapters,
redacted tracing, and experimental evaluation metrics.
