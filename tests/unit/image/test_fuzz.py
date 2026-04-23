from hypothesis import given, settings
from hypothesis import strategies as st

from penampakan.errors import ImageError
from penampakan.image.loader import load_image


@settings(max_examples=10_000, deadline=None)
@given(st.binary(max_size=256))
def test_arbitrary_encoded_input_has_bounded_documented_outcomes(content: bytes) -> None:
    try:
        loaded = load_image(content)
    except ImageError:
        return
    loaded.close()
