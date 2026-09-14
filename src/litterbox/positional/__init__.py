"""Positional encodings, kept explicit and per-layer.

Importing this package registers every strategy, so ``get_positional("rope")``
works without the caller knowing which module defines it — the same contract
``litterbox.model`` keeps for mixers.
"""

from litterbox.positional.base import (
    PositionalEncoding,
    available_positional,
    get_positional,
    register_positional,
)
from litterbox.positional.nope import NoPE
from litterbox.positional.rope import RoPE
from litterbox.positional.sinusoidal import Sinusoidal

__all__ = [
    "NoPE",
    "PositionalEncoding",
    "RoPE",
    "Sinusoidal",
    "available_positional",
    "get_positional",
    "register_positional",
]
