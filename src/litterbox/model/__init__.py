"""Model backbone, block, and mixer registry.

Importing this package pulls in ``litterbox.model.mixers``, which registers every
reference mixer. That means ``get_mixer("gated_deltanet")`` works without the
caller knowing which module defines it — and, more importantly, that an empty
registry is never something you have to debug.
"""

from litterbox.model import mixers  # noqa: F401  (imported for its registrations)
from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import available_mixers, get_mixer, register_mixer

__all__ = [
    "MixerState",
    "TokenMixer",
    "available_mixers",
    "get_mixer",
    "register_mixer",
]
