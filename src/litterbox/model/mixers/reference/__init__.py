"""Reference mixers: pure PyTorch, readable, obviously correct, O(n^2) allowed.

These are the test oracles for anything in ``fast/``.

Importing this package registers every reference mixer, so a config naming one
resolves without the caller knowing which module it lives in.
"""

from litterbox.model.mixers.reference import (  # noqa: F401
    deltanet,
    dsa,
    full_attention,
    gated_deltanet,
    linear_attention,
    mla,
    sliding_window,
)
