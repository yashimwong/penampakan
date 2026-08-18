"""Multi-image placeholder; requires the ordered-root API proposed by specification 10.

The currently shipped client accepts one image per session. Expected output is an
explicit skip message until that dependency lands; this module never simulates support.
"""


def main() -> None:
    print("SKIPPED: multi-image sessions require specification 10 and are not yet available")


if __name__ == "__main__":
    main()
