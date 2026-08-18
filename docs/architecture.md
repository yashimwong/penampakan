# Architecture

Penampakan is async-first. Its synchronous facade delegates to the same async
implementation on one private event-loop thread, so there is one set of
lifecycle, validation, and cancellation rules.

## Layers and dependency direction

| Layer | Responsibility | Depends inward on |
| --- | --- | --- |
| `client` / `sync` | Public one-shot calls, reusable sessions, lifecycle | Session and public protocols |
| `session` / `reasoning` / `tools` | Bounded action loop, trusted context, evidence validation | Domain models, perception routing, asset/observation stores |
| `perception` | Backend selection, normalization, cache keys, single-flight work | Public backend/cache protocols and domain models |
| `image` | Bounded loading, canonicalization, safe transforms, asset lineage | Configuration and domain models |
| `backends` / `llms` | Optional integrations and callable adapters | Public protocols and provider-neutral contracts |
| `models` / `protocols` / `errors` / `config` | Stable contracts and settings | Pydantic and the standard library; no concrete provider/model packages |

Dependencies point toward provider-neutral contracts. The orchestrator does not
import Torch, Transformers, Tesseract wrappers, or provider SDKs. Optional
adapters import their third-party dependency only when constructed or explicitly
initialized, so base-package import remains free of model loading, credential
lookup, network activity, file creation, and telemetry setup.

## Data ownership and lineage

The loader copies or reads a bounded input, applies orientation, normalizes its
mode, encodes a canonical PNG, and hashes that encoding. A session-private asset
store owns the canonical root pixels. Safe transforms create immutable derived
assets with a parent ID, derivation depth, transform descriptor, and content
digest. Identical derived content is reused rather than installed twice.

Backends receive a `BackendImage` containing an immutable asset descriptor and
private canonical bytes. Their untrusted drafts are normalized, then committed
atomically to an append-only observation store. Core code assigns observation
IDs and provenance: tool and capability, backend and model identity, request
hash, parent observations, cache-hit state, and duration. Public results are
immutable snapshots; closing a session releases its private pixels and stores.

## Resource ownership

Ownership follows constructor boundaries, not Python variable ownership:

- An `AsyncPenampakan`/`Penampakan` client owns every registered vision backend,
  its selected cache, and its trace sinks. It closes them; the same backend
  instance cannot be shared across live clients.
- A caller-supplied `ActionPolicy` is caller-owned by default. Set
  `owns_policy=True` to transfer close responsibility to the client.
- A caller-supplied `TextLLM` is caller-owned by default. The convenience
  `llm=` path creates and owns its policy, but closes the LLM only when
  `owns_llm=True`.
- A provider adapter owns an SDK client it constructs. An injected SDK client is
  caller-owned unless that adapter's `owns_client` option transfers ownership.
- A session owns its asset and observation stores and borrows the client's
  router, policy, cache, single-flight coordinator, and sinks.

Client shutdown first drains active operations, then closes sessions,
single-flight work, backends, an owned policy/LLM, the cache, and finally trace
sinks. Cleanup is idempotent and attempts every owned resource. Ordinary close
failures become redacted `AsyncPenampakan.close_warnings`; cancellation and
other base exceptions are retained until cleanup completes and are then
propagated.

Use nested context managers when resources remain caller-owned:

```python
async with llm:
    async with AsyncPenampakan(llm=llm) as vision:
        answer = await vision.ask(image, question)
```

One-shot `inspect` and `ask` calls create and close a private session in all
outcomes. `open_image` returns a reusable session whose caller must close it,
normally with a context manager.
