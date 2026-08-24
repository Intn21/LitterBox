"""The channel mixer interface.

Deliberately thinner than :class:`~litterbox.model.mixers.base.TokenMixer`: no
state, no position, no inference/training distinction. A channel mixer is a pure
function of one token's vector, applied to every token in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from litterbox.model.registry import CHANNEL_MIXERS

if TYPE_CHECKING:
    from torch import Tensor


class ChannelMixer(nn.Module, ABC):
    """Per-token transformation. ``[B, S, D]`` in, ``[B, S, D]`` out."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...


def build_channel_mixer(d_model: int, cfg: dict[str, Any]) -> ChannelMixer:
    """Construct the channel mixer named in config.

    Args:
        d_model: model width.
        cfg: the ``mlp`` block from a model config — a ``type`` key plus whatever
            that implementation takes.
    """
    cfg = dict(cfg)
    name = cfg.pop("type")
    return CHANNEL_MIXERS.get(name)(d_model, **cfg)
