"""Channel mixers — the second half of a block.

A channel mixer transforms each token's vector independently. It never looks at
any other position: hand it one token alone and you get the same answer back.
That is the whole distinction from a token mixer, and it is why this compartment
carries no state and needs no positional information.

Importing this package registers every implementation.
"""

from litterbox.model.channel_mixers import swiglu  # noqa: F401
from litterbox.model.channel_mixers.base import ChannelMixer, build_channel_mixer

__all__ = ["ChannelMixer", "build_channel_mixer"]
