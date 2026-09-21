"""Model backbone, block, and mixer registry.

Importing this package pulls in ``litterbox.model.mixers``, which registers every
reference mixer. That means ``get_mixer("gated_deltanet")`` works without the
caller knowing which module defines it — and, more importantly, that an empty
registry is never something you have to debug.
"""

from litterbox.model import mixers  # noqa: F401  (imported for its registrations)
from litterbox.model.assemble import dense_transformer
from litterbox.model.block import TransformerBlock, scale_residual_projections
from litterbox.model.build import build_block, build_model
from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.mlp import SwiGLU
from litterbox.model.norm import RMSNorm
from litterbox.model.registry import available_mixers, get_mixer, register_mixer

__all__ = [
    "MixerState",
    "RMSNorm",
    "SwiGLU",
    "TokenMixer",
    "TransformerBlock",
    "available_mixers",
    "build_block",
    "build_model",
    "dense_transformer",
    "get_mixer",
    "register_mixer",
    "scale_residual_projections",
]
