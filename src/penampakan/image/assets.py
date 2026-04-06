"""Private session ownership for immutable normalized image assets."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Protocol, cast

from PIL.Image import Image as PillowImage

from penampakan.config import ImageLimits, RunLimits
from penampakan.errors import (
    AssetLimitExceededError,
    AssetNotFoundError,
    DerivationDepthLimitExceededError,
    ImageLimitExceededError,
    InvalidImageError,
    SessionClosedError,
)
from penampakan.models import BackendImage, ImageAsset, TransformDescriptor

OriginalFormat = Literal["PNG", "JPEG", "WEBP"] | None


class LoadedImageLike(Protocol):
    """Normalized loader result accepted by the asset store."""

    image: PillowImage
    canonical_png: bytes
    digest_sha256: str
    original_format: OriginalFormat

    def close(self) -> None:
        """Release the loader-owned image after transfer."""
        ...


@dataclass(frozen=True, slots=True)
class PendingAsset:
    """A fully rendered derivative awaiting transactional store commit."""

    image: PillowImage
    transform: TransformDescriptor

    def close(self) -> None:
        """Release the temporary image."""
        self.image.close()


@dataclass(frozen=True, slots=True)
class AssetCommit:
    """The public asset snapshot and accounting result of one pending asset."""

    asset: ImageAsset
    parent_id: str
    transform: TransformDescriptor
    reused: bool


@dataclass(slots=True)
class _AssetRecord:
    asset: ImageAsset
    image: PillowImage
    canonical_png: bytes

    def close(self) -> None:
        self.image.close()


@dataclass(slots=True)
class _PreparedAsset:
    image: PillowImage
    canonical_png: bytes
    digest_sha256: str
    pending: PendingAsset

    def close(self) -> None:
        self.image.close()


def _validate_dimensions(image: PillowImage, limits: ImageLimits) -> None:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise InvalidImageError()
    if width > limits.max_width or height > limits.max_height:
        raise ImageLimitExceededError()
    if width * height > limits.max_pixels:
        raise ImageLimitExceededError()


def _owned_image(image: PillowImage, limits: ImageLimits) -> PillowImage:
    if not isinstance(image, PillowImage):
        raise TypeError("image must be a Pillow image")
    if image.mode not in {"RGB", "RGBA"}:
        raise InvalidImageError()
    _validate_dimensions(image, limits)
    owned = image.copy()
    owned.info.clear()
    return owned


def canonical_png_bytes(image: PillowImage, limits: ImageLimits | None = None) -> bytes:
    """Encode normalized pixels with the stable core PNG settings."""
    active_limits = limits or ImageLimits()
    owned = _owned_image(image, active_limits)
    output = BytesIO()
    try:
        owned.save(output, format="PNG", optimize=False, compress_level=9)
        return output.getvalue()
    finally:
        output.close()
        owned.close()


def canonical_digest(canonical_png: bytes) -> str:
    """Return the lowercase SHA-256 digest of canonical PNG bytes."""
    return hashlib.sha256(canonical_png).hexdigest()


class AssetStore:
    """Session-private owner of canonical root and derived image assets."""

    def __init__(
        self,
        root_image: PillowImage,
        *,
        original_format: OriginalFormat = None,
        image_limits: ImageLimits | None = None,
        run_limits: RunLimits | None = None,
        canonical_png: bytes | None = None,
        digest_sha256: str | None = None,
    ) -> None:
        self._image_limits = image_limits or ImageLimits()
        self._run_limits = run_limits or RunLimits()
        self._records: dict[str, _AssetRecord] = {}
        self._digest_ids: dict[str, str] = {}
        self._closed = False
        owned = _owned_image(root_image, self._image_limits)
        try:
            encoded = canonical_png_bytes(owned, self._image_limits)
            if canonical_png is not None and canonical_png != encoded:
                raise InvalidImageError()
            content = encoded if canonical_png is None else bytes(canonical_png)
            digest = canonical_digest(content)
            if digest_sha256 is not None and digest_sha256 != digest:
                raise InvalidImageError()
            asset_id = f"img_{digest[:16]}"
            asset = ImageAsset(
                id=asset_id,
                width=owned.width,
                height=owned.height,
                mode=cast(Literal["RGB", "RGBA"], owned.mode),
                mime_type="image/png",
                original_format=original_format,
                digest_sha256=digest,
                parent_id=None,
                derivation_depth=0,
                transform=None,
            )
            self._records[asset_id] = _AssetRecord(asset, owned, content)
            self._digest_ids[digest] = asset_id
            self._root_id = asset_id
        except BaseException:
            owned.close()
            raise

    @classmethod
    def create(
        cls,
        root_image: PillowImage,
        *,
        original_format: OriginalFormat = None,
        image_limits: ImageLimits | None = None,
        run_limits: RunLimits | None = None,
    ) -> AssetStore:
        """Create a store from an already normalized caller-owned image."""
        return cls(
            root_image,
            original_format=original_format,
            image_limits=image_limits,
            run_limits=run_limits,
        )

    @classmethod
    def from_loaded(
        cls,
        loaded: LoadedImageLike,
        image_limits: ImageLimits | None = None,
        run_limits: RunLimits | None = None,
    ) -> AssetStore:
        """Create a store from a normalized loader result."""
        try:
            return cls(
                loaded.image,
                original_format=loaded.original_format,
                image_limits=image_limits,
                run_limits=run_limits,
                canonical_png=loaded.canonical_png,
                digest_sha256=loaded.digest_sha256,
            )
        finally:
            loaded.close()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, asset_id: object) -> bool:
        return isinstance(asset_id, str) and asset_id in self._records

    def __enter__(self) -> AssetStore:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def root_id(self) -> str:
        """Return the stable root asset ID."""
        self._require_open()
        return self._root_id

    @property
    def root(self) -> ImageAsset:
        """Return the immutable public root asset snapshot."""
        return self.snapshot(self.root_id)

    @property
    def derived_count(self) -> int:
        """Return the number of unique committed derived assets."""
        self._require_open()
        return len(self._records) - 1

    @property
    def closed(self) -> bool:
        """Return whether owned resources have been released."""
        return self._closed

    def snapshots(self) -> tuple[ImageAsset, ...]:
        """Return public snapshots in deterministic insertion order."""
        self._require_open()
        return tuple(record.asset for record in self._records.values())

    def snapshot(self, asset_id: str) -> ImageAsset:
        """Return one immutable public asset snapshot."""
        return self._record(asset_id).asset

    def image(self, asset_id: str) -> PillowImage:
        """Return a caller-owned copy of an asset's normalized pixels."""
        record = self._record(asset_id)
        copy = record.image.copy()
        copy.info.clear()
        return copy

    def content(self, asset_id: str) -> bytes:
        """Return immutable canonical PNG bytes for internal backend use."""
        return self._record(asset_id).canonical_png

    def backend_image(self, asset_id: str) -> BackendImage:
        """Create the protected backend-facing image contract."""
        record = self._record(asset_id)
        return BackendImage(asset=record.asset, content=record.canonical_png)

    def ensure_capacity(self, parent_id: str, pending_count: int) -> None:
        """Validate worst-case derivative depth and capacity before rendering."""
        if isinstance(pending_count, bool) or not isinstance(pending_count, int):
            raise TypeError("pending_count must be an integer")
        if pending_count < 0:
            raise ValueError("pending_count cannot be negative")
        parent = self._record(parent_id).asset
        self._validate_depth(parent.derivation_depth + 1)
        if self.derived_count + pending_count > self._run_limits.max_derived_assets:
            raise AssetLimitExceededError()

    def commit(
        self,
        parent_id: str,
        pending_assets: Sequence[PendingAsset],
    ) -> tuple[AssetCommit, ...]:
        """Atomically validate, deduplicate, and commit rendered derivatives."""
        parent = self._record(parent_id).asset
        depth = parent.derivation_depth + 1
        self._validate_depth(depth)
        pending = tuple(pending_assets)
        prepared: list[_PreparedAsset] = []
        try:
            for pending_asset in pending:
                if not isinstance(pending_asset, PendingAsset):
                    raise TypeError("pending assets must be PendingAsset instances")
                image = _owned_image(pending_asset.image, self._image_limits)
                try:
                    content = canonical_png_bytes(image, self._image_limits)
                    prepared.append(
                        _PreparedAsset(
                            image=image,
                            canonical_png=content,
                            digest_sha256=canonical_digest(content),
                            pending=pending_asset,
                        )
                    )
                except BaseException:
                    image.close()
                    raise
            new_digests = {
                prepared_asset.digest_sha256
                for prepared_asset in prepared
                if prepared_asset.digest_sha256 not in self._digest_ids
            }
            if self.derived_count + len(new_digests) > self._run_limits.max_derived_assets:
                raise AssetLimitExceededError()
            return self._commit_prepared(parent_id, depth, prepared)
        except BaseException:
            for prepared_asset in prepared:
                prepared_asset.close()
            raise
        finally:
            for pending_asset in pending:
                if isinstance(pending_asset, PendingAsset):
                    pending_asset.close()

    def close(self) -> None:
        """Idempotently release every privately owned Pillow image."""
        if self._closed:
            return
        self._closed = True
        for record in self._records.values():
            record.close()
        self._records.clear()
        self._digest_ids.clear()

    def _commit_prepared(
        self,
        parent_id: str,
        depth: int,
        prepared: list[_PreparedAsset],
    ) -> tuple[AssetCommit, ...]:
        planned_ids = {
            asset_id: record.asset.digest_sha256 for asset_id, record in self._records.items()
        }
        planned_by_digest = dict(self._digest_ids)
        new_records: dict[str, _AssetRecord] = {}
        results: list[AssetCommit] = []
        retained_images: set[int] = set()
        for prepared_asset in prepared:
            asset_id = planned_by_digest.get(prepared_asset.digest_sha256)
            reused = asset_id is not None
            if asset_id is None:
                asset_id = self._allocate_id(prepared_asset.digest_sha256, planned_ids)
                snapshot = ImageAsset(
                    id=asset_id,
                    width=prepared_asset.image.width,
                    height=prepared_asset.image.height,
                    mode=cast(Literal["RGB", "RGBA"], prepared_asset.image.mode),
                    mime_type="image/png",
                    original_format=None,
                    digest_sha256=prepared_asset.digest_sha256,
                    parent_id=parent_id,
                    derivation_depth=depth,
                    transform=prepared_asset.pending.transform,
                )
                new_records[asset_id] = _AssetRecord(
                    snapshot,
                    prepared_asset.image,
                    prepared_asset.canonical_png,
                )
                retained_images.add(id(prepared_asset.image))
                planned_ids[asset_id] = prepared_asset.digest_sha256
                planned_by_digest[prepared_asset.digest_sha256] = asset_id
            else:
                record = new_records.get(asset_id, self._records.get(asset_id))
                if record is None:
                    raise RuntimeError("asset planning failed")
                snapshot = record.asset
            results.append(
                AssetCommit(
                    asset=snapshot,
                    parent_id=parent_id,
                    transform=prepared_asset.pending.transform,
                    reused=reused,
                )
            )
        self._records.update(new_records)
        self._digest_ids.update(planned_by_digest)
        for prepared_asset in prepared:
            if id(prepared_asset.image) not in retained_images:
                prepared_asset.close()
        return tuple(results)

    @staticmethod
    def _allocate_id(digest: str, assigned: dict[str, str]) -> str:
        for length in range(16, 65):
            candidate = f"img_{digest[:length]}"
            assigned_digest = assigned.get(candidate)
            if assigned_digest is None or assigned_digest == digest:
                return candidate
        raise InvalidImageError()

    def _validate_depth(self, depth: int) -> None:
        if depth > self._run_limits.max_derivation_depth:
            raise DerivationDepthLimitExceededError()

    def _record(self, asset_id: str) -> _AssetRecord:
        self._require_open()
        try:
            return self._records[asset_id]
        except KeyError as error:
            raise AssetNotFoundError() from error

    def _require_open(self) -> None:
        if self._closed:
            raise SessionClosedError()


__all__ = [
    "AssetCommit",
    "AssetStore",
    "LoadedImageLike",
    "PendingAsset",
    "canonical_digest",
    "canonical_png_bytes",
]
