# Product scope

Penampakan is a provider-neutral, inspectable, local-capable visual evidence
layer for text-only language models. It converts a static raster image into
typed observations with backend and asset provenance, and can run a bounded
action loop that returns an answer with citations to those observations.

This is a deliberately narrower claim than “general computer vision” or “better
than every native multimodal model.” Capability and accuracy claims apply only
to the task, dataset, model revision, backend configuration, and resource budget
that were actually evaluated. The current checked-in benchmark measures
[orchestration overhead](../README.md#orchestration-overhead), not answer
accuracy.

## Shipped capabilities

| Capability | Shipped implementation | Notes |
| --- | --- | --- |
| Metadata and dominant colors | Built-in Pillow backend | Available in the base installation. |
| OCR | Optional Tesseract backend | Requires the `ocr` extra and a system Tesseract executable. |
| Captioning and open-vocabulary detection | Optional Transformers backends | Require the `transformers` extra and explicit model revisions for reproducible durable caching. |
| Segmentation | Public request/result contract only | No first-party segmentation backend is shipped. |
| Text-model answering | Callable, OpenAI, Anthropic, and LiteLLM adapters | Provider adapters use separately installed extras. |

Inputs are single, static PNG, JPEG, or WebP images supplied as local paths,
encoded bytes, binary streams, or Pillow images. EXIF orientation is applied and
pixels are normalized to canonical RGB or RGBA assets. Animated images and
remote URLs are rejected. Multi-image questions, video, depth, flow, and
cross-image correspondence are not currently available.

## Non-goals

Penampakan does not:

- execute Python, shell, or another program emitted by a language model;
- make an unsupported perception cue appear through prompting alone;
- treat a caption backend as an OCR, depth, or geometry specialist;
- prove that an answer is true merely because its evidence IDs are valid;
- download remote image URLs on the caller's behalf; or
- enable durable content retention by default.

## Safety, privacy, and evidence

Image dimensions, decoded pixels, input bytes, derived assets, tool calls,
backend calls, LLM calls, context size, and wall-clock time are bounded. Backend
outputs are validated and normalized before they enter the observation store.
The reasoning prompt identifies observation content as untrusted data so text
inside an image is not treated as an instruction.

Trace data is redacted before it reaches a sink. Paths, questions, observation
text, model output, and final answers each require a distinct
`TraceContentPolicy` opt-in; credentials, prompts, headers, raw image content,
and other secrets remain excluded regardless of those flags. See
[Runtime: tracing and retention](runtime.md#tracing-and-retention).

Trace redaction and perception caching are independent controls. The default
cache retains nothing. The optional built-in memory cache retains validated
perception-result JSON only for the life of the client process and clears it on
close. A caller-supplied durable cache may retain OCR text, captions, and other
image-derived observations; selecting or administering that store is the
caller's responsibility. Enabling trace content never enables caching, and
enabling caching never changes trace redaction.

Answer citations are **structural evidence**: the cited observation existed in
the session, was visible to the policy, was not a warning, and belongs to the
input image's lineage. This validation does not prove claim entailment or visual
truth. Penampakan has not independently evaluated evidence entailment unless a
specific published benchmark artifact explicitly says otherwise.

