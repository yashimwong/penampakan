# Runtime behavior

## Inspection and answering

`inspect` executes a caller-specified `InspectionPlan` without an LLM. Its
available-overview mode requests authoritative metadata, dominant colors, and
registered caption/OCR capabilities. Explicit operations may be marked required,
pinned to a backend name, run fail-fast, or target a derived asset in a reusable
session.

`ask` uses a bounded action loop:

1. Load and normalize the image into a session-owned root asset.
2. Gather the configured initial capabilities that registered backends support.
3. Compile observations into a bounded context marked as untrusted visual data.
4. Ask the action policy for exactly one schema-valid tool or answer action.
5. Validate tool names, arguments, and visible asset references before execution.
6. Normalize and commit new assets/observations, then repeat within budget.
7. Validate cited evidence or return an explicit `insufficient_evidence` result.

Invalid model output receives one bounded repair opportunity when the remaining
LLM budget permits it. Repeated identical actions and exhausted soft limits force
an answer-only final call. The library never executes model-generated code.

## Budgets and deadlines

`RunLimits` controls decision steps, LLM reservations, tool calls, actual backend
attempts (including fallbacks), derived assets, derivation depth, parallel tools,
context characters, OCR characters, and overall/component timeouts. Reservations
are atomic and occur before work starts. A per-call `timeout_s` creates one
monotonic deadline covering load, reasoning, backend work, and cleanup-sensitive
control flow; backend and LLM timeouts are capped by the remaining deadline.

Provider retries are separate from orchestrator LLM reservations. A
`RetryPolicy` uses bounded attempts with capped exponential backoff and full
jitter, but the complete provider operation still shares `LLMRequest.timeout_s`.
Every provider attempt and reported token count is reflected in trace metadata.
Adapters retry connection failures, provider timeouts, HTTP 408/429, and 5xx;
other 4xx failures are terminal.

## Tools and backends

The built-in transform tools are crop, row-major tile, right-angle rotate,
contrast enhancement, grayscale conversion, and coordinate-grid overlay. They
create canonical derived assets and enforce asset-count and lineage-depth limits.
Perception tools are registered only when a backend declares support. Metadata
and colors are available from the built-in Pillow backend; metadata is
authoritative, while compatible color backends can participate in normal
preference/fallback routing.

Routing is deterministic: an explicit compatible override, configured
preferences, then registration order. Optional fallback proceeds only through
compatible backends and records every attempt and degradation. A backend's
`max_concurrency` is enforced with a semaphore. Backend output is normalized for
request semantics, size, geometry, confidence, and text bounds before commit.

Model-backed vision descriptors should contain an immutable model revision. If
the adapter cannot resolve one, it emits `unresolved_model_revision` and its
descriptor is not eligible for a durable cross-process cache. Process-local
single-flight deduplication and memory caching remain safe because they do not
claim reproducibility across model generations.

The current API accepts exactly one root image per session. There is therefore
no multi-image ordering or aggregate-image budget to configure yet. Do not build
a collage and describe it as native multi-image evidence; use separate sessions
until an ordered multi-image contract ships.

## Tracing and retention

Every completed operation returns a `RunTrace`. Its `TraceSummary` contains a
UUIDv4 trace ID, UTC start time, duration, LLM/tool/backend/cache/asset counters,
optional provider token totals, and a stop reason. Each ordered `TraceEvent`
contains the same trace ID, a strictly increasing sequence, UTC occurrence time,
an optional duration, an event type, and strict JSON data. The exact public
schema comes from [`TraceEvent` and related models](../src/penampakan/models.py).

Event types cover run start/finish/failure, image load, initial planning, policy
calls, invalid actions, tool/backend calls, cache hits, asset creation,
observation commit, budget stops, and answer validation. Sinks passed through
`trace_sinks=` receive immutable events after redaction; sink failure does not
expose content and is represented as a safe warning. The returned `RunTrace` is
the in-memory record. A custom JSONL or telemetry sink chooses its own durable
storage and access policy and is closed by the client.

Default trace redaction excludes paths, questions, observation text, model
output, and answers. `TraceContentPolicy` opts those categories in independently.
It never permits credentials, authorization data, prompts/messages, raw bytes or
pixels, headers, cookies, environment variables, or secrets.

Caching is a different trust and retention boundary. The default `NullCache`
stores nothing. `CacheSettings(enabled=True)` selects a bounded process-local
LRU containing validated serialized `VisionResult` values, which can include OCR
text and captions; it is cleared on client close. Cache keys bind the canonical
asset digest, request, capability, backend/version/model revision,
preprocessing version, and cache schema version. A caller-supplied cache with
`durable=True` declares cross-process retention and is skipped for unresolved
model revisions. Its persistence, encryption, access controls, TTL, backups, and
deletion behavior are the caller's responsibility.

Trace content flags never enable a cache. Cache settings never weaken trace
redaction.
