import asyncio
from collections.abc import Callable
from typing import cast

import pytest
from pydantic import ValidationError

from penampakan.errors import (
    AssetNotFoundError,
    ObservationNotFoundError,
    SessionClosedError,
)
from penampakan.models import (
    Capability,
    CaptionPayload,
    ObservationDraft,
    Provenance,
    TransformDescriptor,
    TransformPayload,
    VisionResult,
    WarningInfo,
)
from penampakan.perception.store import (
    ObservationRelations,
    ObservationStore,
    ProvenanceSpec,
)

ROOT_ID = "img_0000000000000000"
DERIVED_ID = "img_1111111111111111"
OTHER_ID = "img_2222222222222222"
MISSING_ID = "img_ffffffffffffffff"
OWNED_ASSETS = frozenset({ROOT_ID, DERIVED_ID, OTHER_ID})


def provenance_spec(
    parent_observation_ids: tuple[str, ...] = (),
) -> ProvenanceSpec:
    return ProvenanceSpec(
        tool="describe_image",
        capability=Capability.CAPTION,
        backend_name="scripted.caption",
        backend_version="1.0",
        request_hash="a" * 64,
        duration_ms=12,
        model_id="caption-model",
        model_revision="revision-1",
        parent_observation_ids=parent_observation_ids,
        cache_hit=True,
    )


def caption_draft(text: str) -> ObservationDraft:
    return ObservationDraft(payload=CaptionPayload(text=text))


def vision_result(*drafts: ObservationDraft) -> VisionResult:
    return VisionResult(observations=drafts)


def seeded_store() -> ObservationStore:
    store = ObservationStore(OWNED_ASSETS)
    store.commit_drafts(ROOT_ID, (caption_draft("seed"),), provenance_spec())
    return store


def transform_draft(
    *,
    parent_id: str = ROOT_ID,
    derived_id: str = DERIVED_ID,
) -> ObservationDraft:
    descriptor = TransformDescriptor(
        name="crop",
        parameters={"requested_box": {"x_min": 0.0, "x_max": 0.5}},
    )
    return ObservationDraft(
        payload=TransformPayload(
            derived_asset_id=derived_id,
            parent_asset_id=parent_id,
            transform=descriptor,
        )
    )


def test_empty_and_ordered_batches_allocate_gap_free_ids() -> None:
    store = ObservationStore(OWNED_ASSETS)

    assert store.commit_drafts(ROOT_ID, (), provenance_spec()) == ()
    first = store.commit_result(
        ROOT_ID,
        vision_result(caption_draft("first"), caption_draft("second")),
        provenance_spec().build(),
    )
    second = store.commit_drafts(ROOT_ID, (caption_draft("third"),), provenance_spec())

    assert tuple(item.id for item in first + second) == (
        "obs_000001",
        "obs_000002",
        "obs_000003",
    )
    assert tuple(item.payload.text for item in store.observations) == (
        "first",
        "second",
        "third",
    )
    assert len(store) == 3
    assert "obs_000001" in store
    assert object() not in store
    assert "obs_999999" not in store


def test_commit_copies_provenance_and_draft_values() -> None:
    store = ObservationStore(OWNED_ASSETS)
    source = provenance_spec().build()
    committed = store.commit_drafts(ROOT_ID, (caption_draft("copy"),), source)

    assert committed[0].provenance == source
    assert committed[0].provenance is not source
    assert committed[0].provenance.cache_hit is True
    assert committed[0].provenance.model_revision == "revision-1"


def test_snapshots_and_lookup_are_deep_caller_owned_copies() -> None:
    store = ObservationStore(OWNED_ASSETS)
    committed = store.commit_drafts(
        DERIVED_ID,
        (transform_draft(),),
        provenance_spec(),
    )
    returned = committed[0]
    assert isinstance(returned.payload, TransformPayload)
    returned.payload.transform.parameters["caller_mutation"] = True

    first_snapshot = store.snapshots()[0]
    second_snapshot = store.observations[0]
    lookup = store.get("obs_000001")

    for item in (first_snapshot, second_snapshot, lookup):
        assert isinstance(item.payload, TransformPayload)
        assert "caller_mutation" not in item.payload.transform.parameters
    assert first_snapshot is not second_snapshot
    assert first_snapshot is not lookup


def test_missing_lookup_raises_public_error_without_mutation() -> None:
    store = seeded_store()

    with pytest.raises(ObservationNotFoundError):
        store.get("obs_999999")

    assert tuple(item.id for item in store.snapshots()) == ("obs_000001",)


def test_owned_asset_callback_is_supported() -> None:
    calls: list[str] = []

    def owns(asset_id: str) -> bool:
        calls.append(asset_id)
        return asset_id == ROOT_ID

    store = ObservationStore(owns)
    committed = store.commit_drafts(ROOT_ID, (caption_draft("callback"),), provenance_spec())

    assert committed[0].asset_id == ROOT_ID
    assert calls == [ROOT_ID]


def test_unowned_target_asset_rejects_complete_batch_without_consuming_id() -> None:
    store = ObservationStore(OWNED_ASSETS)

    with pytest.raises(AssetNotFoundError):
        store.commit_drafts(MISSING_ID, (caption_draft("invalid"),), provenance_spec())

    committed = store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())
    assert committed[0].id == "obs_000001"


@pytest.mark.parametrize(
    ("target_id", "draft", "error_type"),
    [
        (
            DERIVED_ID,
            transform_draft(parent_id=MISSING_ID),
            AssetNotFoundError,
        ),
        (
            ROOT_ID,
            transform_draft(derived_id=MISSING_ID),
            AssetNotFoundError,
        ),
        (
            OTHER_ID,
            transform_draft(),
            ValueError,
        ),
    ],
)
def test_transform_assets_must_be_owned_and_target_the_derived_asset(
    target_id: str,
    draft: ObservationDraft,
    error_type: type[Exception],
) -> None:
    store = ObservationStore(OWNED_ASSETS)

    with pytest.raises(error_type):
        store.commit_drafts(target_id, (draft,), provenance_spec())

    assert store.snapshots() == ()
    assert store.commit_drafts(ROOT_ID, (caption_draft("next"),), provenance_spec())[0].id == (
        "obs_000001"
    )


@pytest.mark.parametrize(
    "relations",
    [
        (ObservationRelations(), ObservationRelations()),
        (cast(ObservationRelations, object()),),
        (ObservationRelations(supersedes=("obs_000001", "obs_000001")),),
        (ObservationRelations(contradicts=("obs_000001", "obs_000001")),),
        (
            ObservationRelations(
                supersedes=("obs_000001",),
                contradicts=("obs_000001",),
            ),
        ),
        (ObservationRelations(supersedes=("obs_999999",)),),
        (ObservationRelations(contradicts=("obs_999999",)),),
    ],
)
def test_invalid_relations_reject_batch_without_consuming_id(
    relations: tuple[ObservationRelations, ...],
) -> None:
    store = seeded_store()

    with pytest.raises((TypeError, ValueError, ObservationNotFoundError)):
        store.commit_drafts(
            ROOT_ID,
            (caption_draft("invalid"),),
            provenance_spec(),
            relations=relations,
        )

    committed = store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())
    assert committed[0].id == "obs_000002"


@pytest.mark.parametrize(
    "parent_ids",
    [
        ("obs_000001", "obs_000001"),
        ("obs_999999",),
    ],
)
def test_invalid_provenance_parents_reject_batch_without_consuming_id(
    parent_ids: tuple[str, ...],
) -> None:
    store = seeded_store()

    with pytest.raises((ValueError, ObservationNotFoundError)):
        store.commit_drafts(
            ROOT_ID,
            (caption_draft("invalid"),),
            provenance_spec(parent_ids),
        )

    committed = store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())
    assert committed[0].id == "obs_000002"


def test_valid_relations_and_provenance_parents_are_preserved() -> None:
    store = seeded_store()
    relation = ObservationRelations(
        supersedes=("obs_000001",),
        contradicts=(),
    )

    committed = store.commit_drafts(
        ROOT_ID,
        (caption_draft("replacement"),),
        provenance_spec(("obs_000001",)),
        relations=(relation,),
    )

    assert committed[0].supersedes == ("obs_000001",)
    assert committed[0].contradicts == ()
    assert committed[0].provenance.parent_observation_ids == ("obs_000001",)


def test_draft_warnings_are_preserved() -> None:
    warning = WarningInfo(code="location_unavailable", message="No region was returned.")
    draft = ObservationDraft(payload=CaptionPayload(text="caption"), warnings=(warning,))
    store = ObservationStore(OWNED_ASSETS)

    committed = store.commit_drafts(ROOT_ID, (draft,), provenance_spec())

    assert committed[0].warnings == (warning,)


def test_callback_return_type_is_strict_and_failure_consumes_no_id() -> None:
    return_valid = False

    def owns(asset_id: str) -> bool:
        if return_valid:
            return asset_id in OWNED_ASSETS
        return cast(bool, "yes")

    store = ObservationStore(owns)

    with pytest.raises(TypeError):
        store.commit_drafts(ROOT_ID, (caption_draft("invalid"),), provenance_spec())

    return_valid = True
    committed = store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())
    assert committed[0].id == "obs_000001"


def test_callback_cancellation_before_mutation_consumes_no_id() -> None:
    calls = 0

    def owns(asset_id: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError
        return asset_id in OWNED_ASSETS

    store = ObservationStore(owns)

    with pytest.raises(asyncio.CancelledError):
        store.commit_drafts(ROOT_ID, (caption_draft("cancelled"),), provenance_spec())

    committed = store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())
    assert committed[0].id == "obs_000001"


def test_invalid_later_draft_rolls_back_complete_batch() -> None:
    valid = caption_draft("valid")
    invalid = ObservationDraft.model_construct(payload="invalid")
    result = VisionResult.model_construct(observations=(valid, invalid), warnings=())
    store = ObservationStore(OWNED_ASSETS)

    with pytest.warns(UserWarning), pytest.raises(ValidationError):
        store.commit_result(ROOT_ID, result, provenance_spec())

    committed = store.commit_drafts(ROOT_ID, (caption_draft("after"),), provenance_spec())
    assert committed[0].id == "obs_000001"
    assert len(store) == 1


def test_invalid_constructed_provenance_is_revalidated_before_commit() -> None:
    invalid = Provenance.model_construct(
        tool="not valid",
        capability=Capability.CAPTION,
        backend_name="scripted.caption",
        backend_version="1",
        request_hash="bad",
        parent_observation_ids=(),
        cache_hit=False,
        duration_ms=-1,
    )
    store = ObservationStore(OWNED_ASSETS)

    with pytest.raises(ValidationError):
        store.commit_drafts(ROOT_ID, (caption_draft("invalid"),), invalid)

    assert store.commit_drafts(ROOT_ID, (caption_draft("valid"),), provenance_spec())[0].id == (
        "obs_000001"
    )


def test_wrong_result_provenance_and_draft_types_are_rejected() -> None:
    store = ObservationStore(OWNED_ASSETS)

    with pytest.raises(TypeError):
        store.commit_result(
            ROOT_ID,
            cast(VisionResult, object()),
            provenance_spec(),
        )
    with pytest.raises(TypeError):
        store.commit_drafts(
            ROOT_ID,
            (caption_draft("invalid"),),
            cast(Provenance, object()),
        )
    with pytest.raises(ValidationError):
        store.commit_drafts(
            ROOT_ID,
            (cast(ObservationDraft, object()),),
            provenance_spec(),
        )

    assert store.snapshots() == ()


def test_provenance_spec_build_rejects_invalid_fields() -> None:
    invalid = ProvenanceSpec(
        tool="invalid tool",
        capability=Capability.CAPTION,
        backend_name="scripted.caption",
        backend_version="1",
        request_hash="bad",
        duration_ms=-1,
    )

    with pytest.raises(ValidationError):
        invalid.build()


def test_close_is_idempotent_and_all_public_store_operations_reject_closed_state() -> None:
    store = seeded_store()
    store.close()
    store.close()

    assert store.closed is True
    operations: tuple[Callable[[], object], ...] = (
        lambda: len(store),
        lambda: "obs_000001" in store,
        lambda: store.observations,
        store.snapshots,
        lambda: store.get("obs_000001"),
        lambda: store.commit_result(
            ROOT_ID,
            vision_result(caption_draft("closed")),
            provenance_spec(),
        ),
        lambda: store.commit_drafts(
            ROOT_ID,
            (caption_draft("closed"),),
            provenance_spec(),
        ),
    )
    for operation in operations:
        with pytest.raises(SessionClosedError):
            operation()
