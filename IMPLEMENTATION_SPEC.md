# Penampakan implementation specification

Status: implementation-ready roadmap

Last updated: 2026-08-18

Penampakan is a Python library that turns static visual inputs into bounded,
typed, attributable evidence that a text-only language model can inspect and
reason over. The product goal is not to claim universal superiority over every
native multimodal model. The goal is to be the best provider-neutral,
inspectable, local-capable visual evidence layer for text LLMs, with explicit
budgets, provenance, abstention, and no execution of model-generated code.

The normative improvement roadmap is indexed in `specs/README.md`, and the
individual documents in `specs/` are the implementation requirements. That
directory is a local working area excluded by `.gitignore`, so it is referenced
by path rather than linked: no normative link in this repository may point to a
file that is missing from a fresh checkout.

User-facing documentation describes only shipped behavior and is maintained in:

- [`docs/product.md`](docs/product.md) for product scope, non-goals, safety, and
  privacy;
- [`docs/architecture.md`](docs/architecture.md) for layers, ownership, and
  asset/observation lineage;
- [`docs/contracts.md`](docs/contracts.md) for public API, protocol, wire-schema,
  and compatibility policy;
- [`docs/runtime.md`](docs/runtime.md) for the action loop, budgets, backends,
  trust boundary, tracing, and retention; and
- [`docs/quality.md`](docs/quality.md) for test taxonomy, benchmark
  interpretation, and release gates.

Those guides are explanatory, not a second normative specification. Exact
public signatures and field inventories come from code, docstrings, and schema
snapshots.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY have the meanings from RFC
2119. An implementation MUST NOT silently replace a MUST with another design. If
a requirement proves impractical, record the evidence and decision in
`specs/adr/NNNN-<slug>.md` before deviating.

## Existing architectural contract

Penampakan remains async-first and provider-neutral. It safely loads and
normalizes raster inputs, retains immutable original and derived assets, invokes
registered perception backends through typed capabilities, stores results as
provenance-bearing observations, and runs a bounded JSON action loop in which a
text-only LLM either requests a declared tool or answers with evidence citations.

The implementation MUST preserve:

- no execution of LLM-generated Python or shell code;
- strict, frozen public wire contracts;
- no concrete model imports from the orchestrator;
- bounded tool, backend, model, image, context, and wall-clock use;
- explicit retention controls for every durable content store;
- structural evidence validation and an explicit insufficient-evidence outcome;
- a synchronous facade without weakening async ownership and cancellation rules.

## Definition of success for the roadmap

The roadmap is complete only when:

- the supported-capability benchmark shows quality, abstention risk, latency,
  token, backend-call, and evidence-faithfulness results against blind,
  frozen-perception, agentic, and native-VLM baselines;
- every published result names the exact dataset artifact, prompt, provider
  configuration, backend model revisions, and scorer version;
- multi-image questions and the shipped perception capabilities are representable
  in public contracts without fabricated geometry;
- provider adapters compile the library schema into a tested provider-supported
  subset and expose refusal, truncation, retry, and degradation explicitly;
- real-weight tests exercise coordinate conventions and model provenance;
- traces can be correlated under concurrency without retaining content by default;
- persistent caches are explicit retention opt-ins and cannot silently destroy
  future-version or corrupt data;
- optional Set-of-Mark and evidence-entailment verification are recommended only
  when preregistered evaluations show a favorable quality/risk/cost trade-off;
- documentation, examples, import paths, links, and package metadata are tested
  as release artifacts.
