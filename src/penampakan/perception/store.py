"""Session-local append-only storage for validated observations."""

from __future__ import annotations

from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass, field
from threading import RLock

from penampakan.errors import (
    AssetNotFoundError,
    ObservationNotFoundError,
    SessionClosedError,
)
from penampakan.models import (
    Capability,
    MarkPayload,
    Observation,
    ObservationDraft,
    Provenance,
    TransformPayload,
    VisionResult,
)

AssetOwner = Container[str] | Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class ProvenanceSpec:
    """Validated inputs used to construct one batch's shared provenance."""

    tool: str
    capability: Capability | None
    backend_name: str
    backend_version: str
    request_hash: str
    duration_ms: int
    model_id: str | None = None
    model_revision: str | None = None
    parent_observation_ids: tuple[str, ...] = field(default_factory=tuple)
    cache_hit: bool = False

    def build(self) -> Provenance:
        """Construct strict immutable provenance from the batch inputs."""
        return Provenance(
            tool=self.tool,
            capability=self.capability,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            model_id=self.model_id,
            model_revision=self.model_revision,
            request_hash=self.request_hash,
            parent_observation_ids=self.parent_observation_ids,
            cache_hit=self.cache_hit,
            duration_ms=self.duration_ms,
        )


@dataclass(frozen=True, slots=True)
class ObservationRelations:
    """References from one new observation to previously committed evidence."""

    supersedes: tuple[str, ...] = field(default_factory=tuple)
    contradicts: tuple[str, ...] = field(default_factory=tuple)


class ObservationStore:
    """Append validated observation batches with gap-free session-local IDs."""

    def __init__(self, asset_owner: AssetOwner) -> None:
        if callable(asset_owner):
            self._owns_asset = asset_owner
        else:
            self._owns_asset = lambda asset_id: asset_id in asset_owner
        self._observations: list[Observation] = []
        self._by_id: dict[str, Observation] = {}
        self._next_number = 1
        self._closed = False
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            self._require_open()
            return len(self._observations)

    def __contains__(self, observation_id: object) -> bool:
        with self._lock:
            self._require_open()
            return isinstance(observation_id, str) and observation_id in self._by_id

    @property
    def closed(self) -> bool:
        """Return whether the store has released its session-local state."""
        with self._lock:
            return self._closed

    @property
    def observations(self) -> tuple[Observation, ...]:
        """Return an immutable deep snapshot in append order."""
        return self.snapshots()

    def snapshots(self) -> tuple[Observation, ...]:
        """Return immutable caller-owned snapshots in append order."""
        with self._lock:
            self._require_open()
            return tuple(self._snapshot(item) for item in self._observations)

    def get(self, observation_id: str) -> Observation:
        """Return one caller-owned observation snapshot by session-local ID."""
        with self._lock:
            self._require_open()
            try:
                observation = self._by_id[observation_id]
            except KeyError as error:
                raise ObservationNotFoundError() from error
            return self._snapshot(observation)

    def commit_result(
        self,
        asset_id: str,
        result: VisionResult,
        provenance: Provenance | ProvenanceSpec,
        *,
        relations: Sequence[ObservationRelations] = (),
    ) -> tuple[Observation, ...]:
        """Validate and atomically append every draft in one backend result."""
        validated_result = self._validated_result(result)
        return self._commit(
            asset_id,
            validated_result.observations,
            self._validated_provenance(provenance),
            tuple(relations),
        )

    def commit_drafts(
        self,
        asset_id: str,
        drafts: Sequence[ObservationDraft],
        provenance: Provenance | ProvenanceSpec,
        *,
        relations: Sequence[ObservationRelations] = (),
    ) -> tuple[Observation, ...]:
        """Validate and atomically append a complete ordered draft batch."""
        result = VisionResult(observations=tuple(drafts))
        return self.commit_result(asset_id, result, provenance, relations=relations)

    def close(self) -> None:
        """Release stored references idempotently and reject later operations."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._observations.clear()
            self._by_id.clear()

    def _commit(
        self,
        asset_id: str,
        drafts: tuple[ObservationDraft, ...],
        provenance: Provenance,
        relations: tuple[ObservationRelations, ...],
    ) -> tuple[Observation, ...]:
        with self._lock:
            self._require_open()
            self._require_asset(asset_id)
            resolved_relations = self._validated_relations(relations, len(drafts))
            self._validate_references(provenance.parent_observation_ids)
            for relation in resolved_relations:
                self._validate_relation(relation)
            for draft in drafts:
                self._validate_payload_assets(asset_id, draft)
            prospective = tuple(
                self._build_observation(
                    number=self._next_number + index,
                    asset_id=asset_id,
                    draft=draft,
                    provenance=provenance,
                    relation=resolved_relations[index],
                )
                for index, draft in enumerate(drafts)
            )
            self._append(prospective)
            return tuple(self._snapshot(item) for item in prospective)

    @staticmethod
    def _validated_result(result: VisionResult) -> VisionResult:
        if not isinstance(result, VisionResult):
            raise TypeError("result must be a VisionResult")
        return VisionResult.model_validate(result.model_dump(mode="python"), strict=True)

    @staticmethod
    def _validated_provenance(provenance: Provenance | ProvenanceSpec) -> Provenance:
        if isinstance(provenance, ProvenanceSpec):
            return provenance.build()
        if not isinstance(provenance, Provenance):
            raise TypeError("provenance must be Provenance or ProvenanceSpec")
        return Provenance.model_validate(provenance.model_dump(mode="python"), strict=True)

    def _validated_relations(
        self,
        relations: tuple[ObservationRelations, ...],
        count: int,
    ) -> tuple[ObservationRelations, ...]:
        if not relations:
            return tuple(ObservationRelations() for _ in range(count))
        if len(relations) != count:
            raise ValueError("relations must contain exactly one item per draft")
        for relation in relations:
            if not isinstance(relation, ObservationRelations):
                raise TypeError("relations must contain ObservationRelations values")
        return relations

    def _validate_relation(self, relation: ObservationRelations) -> None:
        if len(relation.supersedes) != len(set(relation.supersedes)):
            raise ValueError("supersedes references must be unique")
        if len(relation.contradicts) != len(set(relation.contradicts)):
            raise ValueError("contradicts references must be unique")
        if set(relation.supersedes).intersection(relation.contradicts):
            raise ValueError(
                "an observation cannot both supersede and contradict the same evidence"
            )
        self._validate_references(relation.supersedes)
        self._validate_references(relation.contradicts)

    def _validate_references(self, observation_ids: tuple[str, ...]) -> None:
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation references must be unique")
        for observation_id in observation_ids:
            if observation_id not in self._by_id:
                raise ObservationNotFoundError()

    def _require_asset(self, asset_id: str) -> None:
        owned = self._owns_asset(asset_id)
        if not isinstance(owned, bool):
            raise TypeError("asset ownership callback must return a boolean")
        if not owned:
            raise AssetNotFoundError()

    def _validate_payload_assets(self, asset_id: str, draft: ObservationDraft) -> None:
        payload = draft.payload
        if not isinstance(payload, (TransformPayload, MarkPayload)):
            return
        self._require_asset(payload.parent_asset_id)
        self._require_asset(payload.derived_asset_id)
        if payload.derived_asset_id != asset_id:
            raise ValueError("transform and mark observations must target their derived asset")
        if isinstance(payload, MarkPayload):
            source_ids = tuple(mark.observation_id for mark in payload.marks)
            self._validate_references(source_ids)
            for mark in payload.marks:
                source = self._by_id[mark.observation_id]
                if source.asset_id != payload.parent_asset_id or source.region != mark.region:
                    raise ValueError("mark references must match their source observations")

    @staticmethod
    def _build_observation(
        *,
        number: int,
        asset_id: str,
        draft: ObservationDraft,
        provenance: Provenance,
        relation: ObservationRelations,
    ) -> Observation:
        return Observation(
            id=f"obs_{number:06d}",
            asset_id=asset_id,
            payload=draft.payload,
            region=draft.region,
            confidence=draft.confidence,
            provenance=provenance,
            supersedes=relation.supersedes,
            contradicts=relation.contradicts,
            warnings=draft.warnings,
        )

    def _append(self, observations: tuple[Observation, ...]) -> None:
        if not observations:
            return
        previous_length = len(self._observations)
        previous_number = self._next_number
        inserted_ids: list[str] = []
        try:
            for observation in observations:
                self._observations.append(observation)
                self._by_id[observation.id] = observation
                inserted_ids.append(observation.id)
            self._next_number += len(observations)
        except BaseException:
            del self._observations[previous_length:]
            for observation_id in inserted_ids:
                self._by_id.pop(observation_id, None)
            self._next_number = previous_number
            raise

    @staticmethod
    def _snapshot(observation: Observation) -> Observation:
        return observation.model_copy(deep=True)

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosedError()


__all__ = [
    "AssetOwner",
    "ObservationRelations",
    "ObservationStore",
    "ProvenanceSpec",
]
