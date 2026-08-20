# Quality and release gates

## Test taxonomy

| Suite | Purpose | External requirements |
| --- | --- | --- |
| Unit | Boundaries, normalization, budgets, routing, lifecycle, retries, and pure metrics | Base/dev dependencies; optional suites are marked |
| Contract | Protocol signatures, strict JSON Schemas, imports, redaction, distribution contents, and provider schema behavior | Provider SDK tests use recording transports and no keys/network |
| End-to-end | Complete deterministic fake-backed workflows and failure outcomes | Offline |
| Integration | Real provider SDK surfaces, Tesseract, Transformers geometry, pinned weights, and end-to-end adapters | Pinned container, executable/model snapshots as applicable |
| Examples/docs | Import safety, offline execution, public snippets, links, and package metadata | Offline examples must not use credentials or network |

The durable perception cache requires SQLite 3.37 or later at runtime for its
`STRICT` tables; an older interpreter build disables the cache instead of
retaining anything, so its suite assumes a supported runtime. The cache's
file-permission and symlink cases skip on platforms without POSIX modes or
symbolic links.

The fast pull-request suite excludes model, OCR, and provider integration markers
and enforces at least 90% package coverage. Ruff, formatting, and strict mypy run
for Python 3.10 through 3.13; Windows runs a base import and smoke suite. Provider
SDK contracts run separately against constrained versions. The scheduled and
release integration workflow uses a pinned container and offline model cache and
fails when a required integration category only skips.

## Benchmark interpretation

The checked-in [metadata manifest](../benchmarks/results/metadata-overhead-2026-08-10.json)
is an orchestration-overhead result. It measures a synthetic local fixture and
metadata decoding; it is neither an answer-quality dataset nor a scorer of model
accuracy. Its contract checks exercise behavior but are reported separately from
latency rankings. Re-run latency comparisons on the same machine and treat short
runs as smoke measurements.

Fake models and deterministic fixtures are valuable for checking harness and
control-flow correctness. Their outputs must never be presented as product
accuracy evidence. A publishable capability result must link an immutable
manifest naming the dataset artifact and revision, scorer/version, exact LLM and
vision model snapshots, provider parameters, prompts, date, repetitions or
resamples, resource budgets, environment, and limitations. Results from unlike
budgets or tasks are not a causal comparison.

Evidence quality has separate dimensions: citation validity, provenance,
claim entailment, localization, and efficiency. `tool_trace_efficiency` measures
provenance/usage diagnostics; it is not a faithfulness or entailment score.
`evidence_region_coverage` reports geometric overlap only. Structural citations
must not be described as entailed unless an independently evaluated entailment
scorer or human protocol established that result.

Set-of-Mark rendering tests establish deterministic geometry, numbering,
deduplication, crowding behavior, alpha preservation, label sanitization, and
canonical PNG output. They do not establish that a visual model can read the
marks or that answer quality improves. Advertising a mark-reference backend
feature requires real weights, pinned revisions, licensed natural fixtures, and
structured scoring of missing, spurious, and correctly attributed mark indices;
ordinary caption text containing a numeral is insufficient. No first-party
backend has passed that gate yet.

The broader Set-of-Mark answer-quality gate also remains open. It requires a
preregistered paired comparison against a no-mark control using the same frozen
source detections, mark-aware backend revision, and resource budget, covering
varied natural and synthetic multi-instance tasks plus accuracy, wrong-answer,
grounding, latency, token, backend-call, derived-asset, and occlusion outcomes.
Until that artifact and its ADR exist, no default prompt or tool change—and no
claim of Set-of-Mark accuracy improvement—is warranted.

## Release gate

Before publishing a release:

1. Run Ruff, formatting, strict mypy, the fast suite with coverage, provider
   contract tests, and required real-dependency categories.
2. Run local Markdown link/anchor checks and documentation snippet/example tests.
3. Build both wheel and sdist, run `twine check`, and inspect their file lists,
   metadata, license, Python classifiers, typed-package marker, extras, and
   project URLs.
4. Confirm base and star imports do not import optional SDKs/models or perform
   credential, network, filesystem-write, logging, or telemetry side effects.
5. Confirm every behavior, compatibility exception, and benchmark claim is
   represented in the [changelog](../CHANGELOG.md) or an immutable manifest.
6. For releases that change a default model, capability, prompt, verification,
   or retention mode, require the corresponding benchmark artifact and ADR.

The version tag must match the package version. Release artifacts are published
only after the verification job succeeds.
