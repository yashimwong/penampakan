import pytest
from PIL import Image

from penampakan.config import RunLimits
from penampakan.errors import (
    AssetLimitExceededError,
    DerivationDepthLimitExceededError,
    SessionClosedError,
)
from penampakan.image.assets import AssetStore, PendingAsset
from penampakan.image.loader import load_image
from penampakan.models import TransformDescriptor
from tests.fixtures.images import encode_image


def transform(name: str = "crop") -> TransformDescriptor:
    return TransformDescriptor(name=name, parameters={"value": 1})


def test_store_owns_root_copy_and_stable_canonical_content() -> None:
    source = Image.new("RGB", (4, 3), "red")

    store = AssetStore.create(source, original_format="PNG")
    source.putpixel((0, 0), (0, 0, 255))
    stored = store.image(store.root_id)

    assert store.root.id.startswith("img_")
    assert store.root.id == f"img_{store.root.digest_sha256[:16]}"
    assert stored.getpixel((0, 0)) == (255, 0, 0)
    assert store.backend_image(store.root_id).content == store.content(store.root_id)
    stored.close()
    store.close()
    source.close()


def test_store_deduplicates_identical_derived_pixels() -> None:
    source = Image.new("RGB", (4, 4), "red")
    store = AssetStore.create(source)
    pending = (
        PendingAsset(Image.new("RGB", (2, 2), "blue"), transform()),
        PendingAsset(Image.new("RGB", (2, 2), "blue"), transform()),
    )

    commits = store.commit(store.root_id, pending)

    assert len(commits) == 2
    assert commits[0].reused is False
    assert commits[1].reused is True
    assert commits[0].asset.id == commits[1].asset.id
    assert store.derived_count == 1
    store.close()
    source.close()


def test_store_rolls_back_batch_when_asset_limit_is_exceeded() -> None:
    source = Image.new("RGB", (4, 4), "red")
    store = AssetStore.create(source, run_limits=RunLimits(max_derived_assets=1))
    pending = (
        PendingAsset(Image.new("RGB", (2, 2), "blue"), transform()),
        PendingAsset(Image.new("RGB", (2, 2), "green"), transform()),
    )

    with pytest.raises(AssetLimitExceededError):
        store.commit(store.root_id, pending)

    assert store.snapshots() == (store.root,)
    store.close()
    source.close()


def test_store_enforces_derivation_depth() -> None:
    source = Image.new("RGB", (4, 4), "red")
    store = AssetStore.create(source, run_limits=RunLimits(max_derivation_depth=1))
    first = store.commit(
        store.root_id,
        (PendingAsset(Image.new("RGB", (2, 2), "blue"), transform()),),
    )[0]

    with pytest.raises(DerivationDepthLimitExceededError):
        store.commit(
            first.asset.id,
            (PendingAsset(Image.new("RGB", (1, 1), "green"), transform()),),
        )

    store.close()
    source.close()


def test_store_close_is_idempotent_and_rejects_access() -> None:
    source = Image.new("RGB", (1, 1), "white")
    store = AssetStore.create(source)

    store.close()
    store.close()

    with pytest.raises(SessionClosedError):
        store.snapshots()
    source.close()


def test_store_consumes_loaded_image_ownership() -> None:
    source = Image.new("RGB", (2, 2), "red")
    loaded = load_image(encode_image(source))

    store = AssetStore.from_loaded(loaded)

    with pytest.raises(ValueError):
        loaded.image.getbbox()
    assert store.root.original_format == "PNG"
    store.close()
    source.close()
