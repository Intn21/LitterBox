"""TransformerBlock — the mixer is injected, never constructed here.

The block does exactly two things: let tokens exchange information (the token
mixer), then let each token transform on its own (the channel mixer). Both are
wrapped in a residual connection and preceded by a norm.

Nothing in this file knows or asks what kind of mixer it is holding. The moment
it does — an ``isinstance`` check, a special case for one family's extra
projection — the compartment has stopped being a compartment. Every mixer is
``[B, S, D]`` in, ``[B, S, D]`` out, and owns whatever projections it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.model.channel_mixers.base import ChannelMixer
    from litterbox.model.mixers.base import MixerState, TokenMixer


class TransformerBlock(nn.Module):
    """One block: ``x + mixer(norm(x))``, then ``x + mlp(norm(x))``.

    Pre-norm, sequential wiring. Both are choices rather than laws — post-norm
    and parallel (GPT-J style) wiring are the obvious next variants, and they
    belong here as config rather than as new block classes.
    """

    def __init__(
        self,
        token_mixer: TokenMixer,
        channel_mixer: ChannelMixer,
        norm_1: nn.Module,
        norm_2: nn.Module,
    ) -> None:
        super().__init__()
        self.token_mixer = token_mixer
        self.channel_mixer = channel_mixer
        self.norm_1 = norm_1
        self.norm_2 = norm_2

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        mixed, state = self.token_mixer(self.norm_1(x), state, pos_offset)
        x = x + mixed
        x = x + self.channel_mixer(self.norm_2(x))
        return x, state
