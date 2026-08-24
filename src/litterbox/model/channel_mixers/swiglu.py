"""SwiGLU and a plain GELU MLP.

Two implementations, because a compartment with one thing behind it has not been
tested as a compartment.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from litterbox.model.channel_mixers.base import ChannelMixer
from litterbox.model.registry import CHANNEL_MIXERS


def _round_to(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


@CHANNEL_MIXERS.register("swiglu")
class SwiGLU(ChannelMixer):
    """Gated MLP: ``down(silu(gate(x)) * up(x))``.

    The gate branch decides how much of the up branch survives, per channel — a
    learned, input-dependent filter rather than a fixed nonlinearity.

    Note the parameter count. SwiGLU uses three matrices where a plain MLP uses
    two, so ``hidden_mult: 4`` here is roughly 1.5x the parameters of a GELU MLP
    at the same multiplier. Llama compensates by using 8/3 instead of 4. This
    implementation takes ``hidden_mult`` literally and leaves the choice to the
    config, so that two configs claiming the same multiplier mean the same thing.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_mult: float = 4.0,
        hidden_dim: int | None = None,
        multiple_of: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else int(hidden_mult * d_model)
        hidden = _round_to(hidden, multiple_of)
        self.hidden_dim = hidden

        self.gate_proj = nn.Linear(d_model, hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@CHANNEL_MIXERS.register("mlp")
class MLP(ChannelMixer):
    """Plain two-matrix MLP with GELU. The pre-gating baseline."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_mult: float = 4.0,
        hidden_dim: int | None = None,
        multiple_of: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else int(hidden_mult * d_model)
        hidden = _round_to(hidden, multiple_of)
        self.hidden_dim = hidden

        self.up_proj = nn.Linear(d_model, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))
