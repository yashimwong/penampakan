from datetime import datetime, timezone
from inspect import Parameter, iscoroutinefunction, signature
from uuid import UUID

import pytest
from pydantic import ValidationError

from penampakan.models import (
    AnswerAction,
    BackendDescriptor,
    BackendImage,
    Capability,
    CapabilityDescriptor,
    ImageAsset,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    MetadataRequest,
    PolicyAction,
    PolicyInput,
    RemainingBudget,
    TraceEvent,
    VisionRequest,
    VisionResult,
)
from penampakan.perception.cache import is_durable_cache
from penampakan.protocols import ActionPolicy, Cache, TextLLM, TraceSink, VisionBackend


class RecordingLLM:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text='{"type":"answer"}', model_id="model-v1")


class EmptyExtension:
    pass


class RecordingBackend:
    def __init__(self, descriptor: BackendDescriptor, result: VisionResult) -> None:
        self._descriptor = descriptor
        self.result = result
        self.calls: list[tuple[BackendImage, VisionRequest]] = []
        self.close_count = 0

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def supports(self, request: VisionRequest) -> bool:
        return request.capability is Capability.METADATA

    async def analyze(self, image: BackendImage, request: VisionRequest) -> VisionResult:
        self.calls.append((image, request))
        return self.result

    async def aclose(self) -> None:
        self.close_count += 1


class FixedPolicy:
    def __init__(self, action: PolicyAction) -> None:
        self.action = action
        self.inputs: list[PolicyInput] = []

    async def next_action(self, input: PolicyInput) -> PolicyAction:
        self.inputs.append(input)
        return self.action


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.sizes: dict[str, int] = {}
        self.close_count = 0

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes, *, size: int) -> None:
        self.values[key] = value
        self.sizes[key] = size

    async def aclose(self) -> None:
        self.close_count += 1


class RecordingTraceSink:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.close_count = 0

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def aclose(self) -> None:
        self.close_count += 1


def make_asset() -> ImageAsset:
    return ImageAsset(
        id="img_0123456789abcdef",
        width=1,
        height=1,
        mode="RGB",
        mime_type="image/png",
        original_format="PNG",
        digest_sha256="a" * 64,
        parent_id=None,
        derivation_depth=0,
        transform=None,
    )


def make_descriptor() -> BackendDescriptor:
    return BackendDescriptor(
        name="example.metadata",
        version="1.0.0",
        capabilities=(CapabilityDescriptor(capability=Capability.METADATA),),
    )


def make_policy_input() -> PolicyInput:
    return PolicyInput(
        question="What is visible?",
        context="No semantic observations are available.",
        tools=(),
        prior_actions=(),
        remaining=RemainingBudget(
            steps=1,
            llm_calls=1,
            tool_calls=0,
            backend_calls=0,
            derived_assets=0,
            derivation_depth=0,
            context_chars=1024,
            remaining_time_s=1.0,
        ),
    )


async def exercise_policy(policy: ActionPolicy, input: PolicyInput) -> PolicyAction:
    return await policy.next_action(input)


async def exercise_cache(cache: Cache) -> None:
    await cache.set("cache-key", b"value", size=5)
    assert await cache.get("cache-key") == b"value"
    assert await cache.get("missing-key") is None
    await cache.aclose()


async def exercise_trace_sink(sink: TraceSink, event: TraceEvent) -> None:
    await sink.emit(event)
    await sink.aclose()


def test_protocol_method_shapes_match_the_public_contract() -> None:
    assert isinstance(VisionBackend.descriptor, property)
    assert iscoroutinefunction(TextLLM.complete)
    assert iscoroutinefunction(VisionBackend.analyze)
    assert iscoroutinefunction(VisionBackend.aclose)
    assert iscoroutinefunction(ActionPolicy.next_action)
    assert iscoroutinefunction(Cache.get)
    assert iscoroutinefunction(Cache.set)
    assert iscoroutinefunction(Cache.aclose)
    assert iscoroutinefunction(TraceSink.emit)
    assert iscoroutinefunction(TraceSink.aclose)
    assert tuple(signature(TextLLM.complete).parameters) == ("self", "request")
    assert tuple(signature(VisionBackend.supports).parameters) == ("self", "request")
    assert tuple(signature(VisionBackend.analyze).parameters) == ("self", "image", "request")
    assert tuple(signature(ActionPolicy.next_action).parameters) == ("self", "input")
    assert tuple(signature(Cache.get).parameters) == ("self", "key")
    assert tuple(signature(Cache.set).parameters) == ("self", "key", "value", "size")
    assert signature(Cache.set).parameters["size"].kind is Parameter.KEYWORD_ONLY
    assert tuple(signature(TraceSink.emit).parameters) == ("self", "event")


async def test_text_llm_is_runtime_checkable_and_completes_request() -> None:
    llm = RecordingLLM()
    request = LLMRequest(
        messages=(Message(role=MessageRole.SYSTEM, content="Return one JSON object."),),
        response_json_schema={
            "additionalProperties": False,
            "properties": {},
            "type": "object",
        },
    )

    assert isinstance(llm, TextLLM)
    assert not isinstance(EmptyExtension(), TextLLM)

    response = await llm.complete(request)

    assert response == LLMResponse(text='{"type":"answer"}', model_id="model-v1")
    assert llm.requests == [request]


async def test_vision_backend_is_runtime_checkable_and_analyzes_image() -> None:
    result = VisionResult(observations=())
    backend = RecordingBackend(make_descriptor(), result)
    image = BackendImage(asset=make_asset(), content=b"canonical-png")
    request = MetadataRequest()

    assert isinstance(backend, VisionBackend)
    assert not isinstance(EmptyExtension(), VisionBackend)
    assert backend.descriptor is backend.descriptor
    assert backend.supports(request)
    assert await backend.analyze(image, request) is result

    await backend.aclose()

    assert backend.calls == [(image, request)]
    assert backend.close_count == 1


async def test_action_policy_uses_typed_input_and_action() -> None:
    action = AnswerAction(
        status="insufficient_evidence",
        answer="The requested fact is not visible.",
    )
    policy = FixedPolicy(action)
    input = make_policy_input()

    assert await exercise_policy(policy, input) is action
    assert policy.inputs == [input]


async def test_cache_uses_bytes_and_keyword_only_accounted_size() -> None:
    cache = MemoryCache()

    await exercise_cache(cache)

    assert cache.values == {"cache-key": b"value"}
    assert cache.sizes == {"cache-key": 5}
    assert cache.close_count == 1
    assert is_durable_cache(cache) is False


async def test_trace_sink_receives_typed_redacted_event() -> None:
    event = TraceEvent(
        trace_id=UUID("00000000-0000-0000-0000-000000000001"),
        sequence=1,
        event_type="backend.completed",
        occurred_at=datetime(2026, 2, 10, 9, 0, tzinfo=timezone.utc),
        duration_ms=5,
        data={"backend": "example.metadata"},
    )
    sink = RecordingTraceSink()

    await exercise_trace_sink(sink, event)

    assert sink.events == [event]
    assert sink.close_count == 1


def test_backend_image_content_is_absent_from_repr_and_serialization() -> None:
    secret = b"backend-image-secret-sentinel"
    image = BackendImage(asset=make_asset(), content=secret)

    assert image.content == secret
    assert "backend-image-secret-sentinel" not in repr(image)
    assert "backend-image-secret-sentinel" not in str(image)
    assert "content" not in image.model_dump()
    assert "content" not in image.model_dump(mode="json")
    assert "content" not in image.model_dump_json()

    with pytest.raises(ValidationError):
        image.content = b"replacement"


def test_backend_image_content_field_is_protected_by_default() -> None:
    content_field = BackendImage.model_fields["content"]

    assert content_field.repr is False
    assert content_field.exclude is True
