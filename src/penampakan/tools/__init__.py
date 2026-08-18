"""Built-in perception and image transformation tools.

This package intentionally exports no names of its own. Its tool registration
helpers are wired by the client during construction and carry no compatibility
promise; applications extend the surface with their own ``VisionBackend``
implementations instead.
"""

__all__: tuple[str, ...] = ()
