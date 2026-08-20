import json
from uuid import UUID

from penampakan.config import TraceContentPolicy
from penampakan.models import TraceEvent
from penampakan.tracing import TraceBuilder, redact_trace_data


class CapturingSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.close_count = 0

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        self.close_count += 1


def sentinel_payload() -> tuple[dict[str, object], tuple[str, ...]]:
    sentinels = (
        "path-secret-sentinel",
        "question-secret-sentinel",
        "ocr-secret-sentinel",
        "caption-secret-sentinel",
        "prompt-secret-sentinel",
        "answer-secret-sentinel",
        "bytes-secret-sentinel",
        "backend-secret-sentinel",
        "api-key-secret-sentinel",
        "tool-arguments-secret-sentinel",
        "model-output-secret-sentinel",
        "verifier-reason-secret-sentinel",
    )
    payload: dict[str, object] = {
        "path": sentinels[0],
        "question": sentinels[1],
        "ocr_text": sentinels[2],
        "caption": sentinels[3],
        "prompt": sentinels[4],
        "answer": sentinels[5],
        "image_bytes": sentinels[6].encode(),
        "backend_exception": RuntimeError(sentinels[7]),
        "api_key": sentinels[8],
        "tool_arguments": {"query": sentinels[9]},
        "raw_model_output": sentinels[10],
        "verifier_reason": sentinels[11],
        "nested": {
            "source_path": sentinels[0],
            "user_question": sentinels[1],
            "raw_text": sentinels[2],
            "final_answer": sentinels[5],
            "credentials": sentinels[8],
        },
        "safe": {
            "backend_name": "example.backend",
            "asset_id": "img_0123456789abcdef",
            "count": 2,
        },
    }
    return payload, sentinels


def assert_sentinels_absent(value: object, sentinels: tuple[str, ...]) -> None:
    serialized = json.dumps(value, default=str, sort_keys=True)
    representation = repr(value)
    for sentinel in sentinels:
        assert sentinel not in serialized
        assert sentinel not in representation


def test_default_redaction_removes_every_sensitive_content_category() -> None:
    payload, sentinels = sentinel_payload()

    redacted = redact_trace_data(payload)

    assert redacted["safe"] == {
        "asset_id": "img_0123456789abcdef",
        "backend_name": "example.backend",
        "count": 2,
    }
    assert redacted["backend_exception"] == {"error_type": "RuntimeError"}
    assert redacted["nested"] == {}
    assert_sentinels_absent(redacted, sentinels)


async def test_trace_events_sinks_and_results_receive_only_redacted_data() -> None:
    payload, sentinels = sentinel_payload()
    sink = CapturingSink()
    builder = TraceBuilder(
        trace_id=UUID("00000000-0000-4000-8000-000000000010"),
        sinks=(sink,),
    )

    await builder.start(payload)
    await builder.emit("backend_call_finished", payload)
    trace = await builder.fail(RuntimeError(sentinels[7]), payload)

    assert sink.events == list(trace.events)
    assert_sentinels_absent(trace.model_dump(mode="json"), sentinels)
    assert_sentinels_absent(sink.events, sentinels)
    assert_sentinels_absent(builder.warnings, sentinels)

    await builder.aclose()

    assert sink.close_count == 1


def test_content_opt_ins_never_allow_credentials_prompts_or_image_bytes() -> None:
    payload, sentinels = sentinel_payload()
    policy = TraceContentPolicy(
        include_paths=True,
        include_questions=True,
        include_observation_text=True,
        include_model_output=True,
        include_answers=True,
    )

    redacted = redact_trace_data(payload, policy)

    assert redacted["path"] == sentinels[0]
    assert redacted["question"] == sentinels[1]
    assert redacted["ocr_text"] == sentinels[2]
    assert redacted["caption"] == sentinels[3]
    assert redacted["answer"] == sentinels[5]
    assert redacted["raw_model_output"] == sentinels[10]
    assert redacted["verifier_reason"] == sentinels[11]
    assert "prompt" not in redacted
    assert "image_bytes" not in redacted
    assert "api_key" not in redacted
    assert "tool_arguments" not in redacted
    assert sentinels[4] not in repr(redacted)
    assert sentinels[6] not in repr(redacted)
    assert sentinels[8] not in repr(redacted)
    assert sentinels[9] not in repr(redacted)
