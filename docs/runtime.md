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
descriptor is not eligible for a durable cross-process cache. A caller-supplied
cache counts as durable unless it declares `durable = False`, so an
undeclared cache is bypassed rather than trusted. Process-local single-flight
deduplication and the shipped `NullCache`/`MemoryLRUCache` remain safe because they
declare themselves non-durable and claim no reproducibility across model
generations; the shipped `SQLiteCache` declares itself durable and is bypassed for
an unresolved revision.

The current API accepts exactly one root image per session. There is therefore
no multi-image ordering or aggregate-image budget to configure yet. Do not build
a collage and describe it as native multi-image evidence; use separate sessions
until an ordered multi-image contract ships.

## Tracing and retention

Every completed operation returns a `RunTrace`. Its `TraceSummary` contains a
UUIDv4 trace ID, UTC start time, duration, LLM/tool/backend/cache/asset counters,
optional provider token totals, and a stop reason. Newly emitted `TraceEvent`s
use schema version 2 and contain the same trace ID, a strictly increasing
sequence, UTC occurrence time, an optional duration, opaque invocation and
parent-invocation IDs, an event type, and strict JSON data. Policy, tool,
backend, and verification starts have exactly one correlated finish. Correlation
uses IDs, never sequence adjacency. Terminal events repeat all summary counters,
including token totals, so streaming sinks are self-contained. The exact public
schema comes from [`TraceEvent` and related models](../src/penampakan/models.py).

An absent `schema_version` parses as legacy v1. Only events explicitly marked v2
have correlation guarantees. Readers must reject unsupported future versions or
retain their JSON opaquely; they must not reinterpret them as v2.

Event types cover run start/finish/failure, image load, initial planning, policy
calls, invalid actions, tool/backend calls, cache hits, asset creation,
observation commit, budget stops, and answer validation. Sinks passed through
`trace_sinks=` receive immutable events after redaction; sink failure does not
expose content and is represented as a safe warning. Caller-supplied sinks are
caller-owned and remain open when the client closes. Pass
`owns_trace_sinks=True` only when the client should drain and close them after
all sessions and other owned resources.

Three destinations ship with the library:

- `InMemoryTraceSink` retains bounded whole completed runs for tests and notebooks.
- `JsonlTraceSink` uses one bounded queue and writer task, private `0700`/`0600`
  artifacts where POSIX modes apply, bounded rotation, and optional fsync. Its
  default `drop_new` overflow never waits and exposes loss through `stats()` and
  a safe warning; `block` explicitly enables backpressure. It is safe for
  concurrent sessions in one process only. Sharing its path between processes
  is unsupported because it provides no cross-process lock.
- `OpenTelemetryTraceSink` requires the `opentelemetry` extra and an injected
  tracer provider. It never configures global telemetry state. CI exercises API
  and SDK 1.44.0 with semantic conventions 0.65b0.

JSONL is durable plaintext retention and is not encryption. Operators choose its
access controls, backups, rotation, and deletion policy. Symbolic links in any
configured path component are refused by default; opting in accepts the
platform's link race limitations.

Default trace redaction excludes paths, questions, observation text, model
output, and answers. `TraceContentPolicy` opts those categories in independently.
It never permits credentials, authorization data, prompts/messages, raw bytes or
pixels, headers, cookies, environment variables, or secrets.

Trace content flags never enable a cache. Cache settings never weaken trace
redaction. Caching is a separate trust and retention boundary, described below.

## Perception caching

`CacheSettings.mode` selects retention explicitly and defaults to `off`:

| Mode | Implementation | Retention |
| --- | --- | --- |
| `off` | `NullCache` | Nothing is stored. |
| `memory` | `MemoryLRUCache` | Bounded process-local LRU, cleared when the client closes. |
| `sqlite` | `SQLiteCache` | Durable on disk at the required `path`, until entries expire or are cleared. |

`path` is required for `sqlite` and rejected for the other modes; `ttl_s`,
`busy_timeout_s`, and `allow_symlink` apply to `sqlite` only. `max_entries` and
`max_bytes` bound both retaining modes. A cache passed as `cache=` takes
precedence over these settings, and its retention behavior is the caller's
responsibility.

### What is stored

One entry is the validated serialized `VisionResult` of one perception call:
strict UTF-8 JSON, retained as the exact bytes that were produced. It can contain
OCR text, captions, detected labels, and other text read out of a user image, so
it is at least as sensitive as the image it was derived from.

The entry key is a SHA-256 digest over the canonical asset digest, the normalized
request, its capability, the backend name, backend version, model ID, exact model
revision, the preprocessing version, and the cache schema version. Changing any one
dimension yields a different key and therefore a miss. A durable cache is bypassed
entirely for a backend whose model revision is unresolved, and a hit reconstructs
provenance only after the key is recomputed for the descriptor about to be
credited, so `cache_hit=True` never masks backend attribution.

A durable row additionally holds the key, the byte size the cache verified itself,
the creation time, an approximate last-access time, and an absolute expiry time
when a TTL is configured. A `meta` table records the database schema version, the
cache schema version, and the value encoding version.

### TTL, recency, and eviction

TTL and watermarks below describe `mode="sqlite"`. The memory cache has no TTL,
evicts strictly least-recently-used down to `max_entries`/`max_bytes` on every
write, and silently declines a value larger than `max_bytes`.

`ttl_s` is absolute from write time: `expires_at = created_at + ttl_s`, compared
against wall-clock UTC epoch time because the database is shared between
processes. Replacing a value restarts its lifetime. Expired entries are misses and
are removed lazily rather than on a timer: a read that lands on an expired entry, a
write, and `prune()` each delete every entry that has already expired. Without
`ttl_s`, entries persist until they are evicted or cleared.

Recency is deliberately approximate. Updating the access time on every read would
turn every reader into a writer, so a hit refreshes it at most once per touch
interval — 60 seconds, a `SQLiteCache` constructor argument that `CacheSettings`
does not expose. Eviction orders candidates by that approximate recency, so an
entry read within the current interval can still be evicted ahead of a slightly
older one.

`max_entries` and `max_bytes` are high watermarks. A write that crosses either one
is accepted first and then evicts, inside the same transaction and oldest first,
until both the entry count and the byte total are at or below the low watermark of
90% of the configured limit; the entry just written is never the victim. The cache
therefore exceeds its configured limit by at most one accepted entry and settles
below the limit rather than at it. A single value larger than `max_bytes` is never
stored — the write is a no-op that records `cache_value_too_large`, because
accepting it would evict everything else and still not fit.

### Filesystem artifacts and permissions

A durable cache path must be absolute and literal. A relative path, an unexpanded
`~`, `$VAR` or `%VAR%`, a glob character, or a NUL byte is refused instead of
resolved, so the cache never derives a destructive target of its own.

At that path SQLite maintains the database file plus `-wal` and `-shm` sidecars
while it is open, and any quarantined predecessor stays beside them until an
operator removes it.

Where the platform has POSIX permissions, a parent directory Penampakan creates is
set to `0700`, and a database file Penampakan creates is opened `0600` before
SQLite touches it, so it never exists briefly under a wider umask. Sidecars
present once the database is open are tightened to `0600`. A directory or database
file that already existed is not modified — its mode is the operator's decision —
but permissions broader than owner-only on any of these artifacts are reported as
`cache_directory_permissions` or `cache_file_permissions` in
`SQLiteCache.warnings`, which are readable on the instance and are not raised.
Platforms without POSIX modes get no permission handling at all.

A symlinked cache path is refused: the instance disables itself with
`cache_symlink_rejected`. `allow_symlink=True` is an advanced opt-in with an
unavoidable time-of-check/time-of-use gap — the check happens before the database
is opened, so a path replaced with a symlink in between is followed. Use it only
where no other user can write the containing directory.

Cache content is not encrypted. Values are plain JSON bytes in an ordinary SQLite
file; anyone who can read the file can read the derived text. Put the cache on an
encrypted filesystem or volume when at-rest confidentiality is required.

`clear()` removes every logical entry in one transaction. That is reclamation, not
secure erasure: it promises nothing about data left in database pages, in the
write-ahead log, in blocks retained by SSD wear levelling, or in backups and
filesystem snapshots. No `VACUUM` is performed, and running one would still be
reclamation rather than erasure. Destroying the underlying media or volume is the
only erasure this feature can point to.

### Contention, degradation, and disablement

One dedicated worker thread creates, owns, and closes the SQLite connection and
drains an ordered queue; the event loop never touches the connection. The worker
enables WAL and `synchronous=NORMAL`, applies a busy timeout derived from
`busy_timeout_s`, runs each operation in a short explicit transaction, and retries
`BUSY`/`LOCKED` with bounded jittered backoff until that deadline. WAL and a busy
timeout reduce contention; they do not guarantee the absence of lock errors or
writer starvation. Exhausting the deadline is a bounded outcome rather than a run
failure: the session-facing surface degrades to a miss or a no-op.

A cache failure never fails a vision run. It produces one redacted
`cache_operation_failed` warning for the perception call, one trace event of the
same name, and increments `TraceSummary.cache_failures`. A retained value that no
longer validates is ignored with `invalid_cache_entry`.

A runtime path or data problem disables the instance instead of breaking client
construction; a programmer error in the settings themselves still raises. A
disabled `SQLiteCache` remains a valid cache — every read misses, every write is a
no-op, close works — with `available` false and `status` carrying the reason:

| `status.code` | Meaning |
| --- | --- |
| `sqlite_version_unsupported` | The runtime is older than SQLite 3.37, so the `STRICT` tables this schema depends on are unavailable. The cache disables itself rather than dropping `STRICT` and claiming equivalent validation. |
| `cache_schema_too_new` | The database was written by a newer schema version. It is opened for no data and left untouched, so an older build never drops future data. |
| `cache_path_unusable` | The path is unusable: not absolute or otherwise unresolved, a directory, uncreatable, unreadable, or locked past the deadline. |
| `cache_symlink_rejected` | The path is a symbolic link and `allow_symlink` is false. |
| `cache_quarantine_failed` | An unusable file could not be moved aside, so it was left exactly as it was and nothing was overwritten. |

An older, unrecognized, or corrupt database is quarantined rather than deleted. The
file and its sidecars are atomically renamed to `<name>.superseded-<UTC timestamp>-<nonce>`
or `<name>.corrupt-<UTC timestamp>-<nonce>` and a fresh database is created beside
them. No migration is defined for a released layout yet, and the quarantined file is
preserved until an operator removes it.

The client does not inspect `available` or `status`, so a disabled durable cache is
indistinguishable from one that never hits. An operator who needs that signal
constructs the `SQLiteCache` directly and passes it as `cache=`.

### Administration

`NullCache`, `MemoryLRUCache`, and `SQLiteCache` all satisfy `ManagedCache`.
`stats()` returns a `CacheStats` snapshot derived transactionally from verified
stored sizes rather than caller-supplied accounting, `clear()` removes every entry,
and `prune()` drops expired and over-watermark entries and reports what it
discarded in `removed_entries`/`removed_bytes`. A durable `stats()` counts live
entries only, so entries that have expired but not yet been deleted are already
excluded from it while still occupying the file. Unlike the session-facing `Cache`
surface, these three raise instead of degrading — a `PenampakanError` carrying
`cache_closed`, the disable reason code, or a retryable `cache_unavailable` —
because an operator who asked a direct question is misled by silence. Because the
database is shared, a durable cache can be administered from a separate process by
constructing a `SQLiteCache` on the same path.
