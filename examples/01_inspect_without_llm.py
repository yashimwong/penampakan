"""Inspect a generated image with the base install; prints dimensions and payload types."""

from PIL import Image

from penampakan import (
    ColorsRequest,
    InspectionOperation,
    InspectionPlan,
    MetadataRequest,
    Penampakan,
)


def main() -> None:
    plan = InspectionPlan(
        operations=(
            InspectionOperation(request=MetadataRequest(), required=True),
            InspectionOperation(request=ColorsRequest(count=3), required=True),
        ),
        include_available_overview=False,
    )
    with Image.new("RGB", (64, 40), "tomato") as image, Penampakan() as vision:
        result = vision.inspect(image, plan)
    payloads = ",".join(item.payload.type for item in result.observations)
    print(f"image={result.root_asset.width}x{result.root_asset.height}")
    print(f"observations={payloads}")


if __name__ == "__main__":
    main()
