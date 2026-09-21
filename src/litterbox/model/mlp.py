"""Channel mixers: the feed-forward half of a block. SwiGLU today; a second one
joins it in ROADMAP step 5.

A block has two halves. The token mixer (attention) moves information *between
positions*. The channel mixer moves information *between channels, within one
position*: it is applied to every token separately and identically, and no
token ever sees another through it. Attention gathers; this layer thinks about
what was gathered. It also holds most of a transformer's parameters.

The classic version is two matrices with a nonlinearity between them:

    y = W_down · act(W_up · x)

SwiGLU (Shazeer, 2020) replaces the fixed nonlinearity with a learned *gate*.
Two projections go up instead of one. One is the content; the other passes
through SiLU and decides, channel by channel, how much of that content gets
through:

    gate    = silu(W_gate · x)          how open is each hidden channel?
    content = W_up · x                  what is in each hidden channel?
    y       = W_down · (gate * content)

    silu(z) = z · sigmoid(z)

A plain activation can only switch a channel based on that channel's own
value. A gate lets one learned view of the input control another, which is a
multiplicative interaction a plain MLP cannot express in a single layer.
Shazeer's paper offers no theory for why it helps — "we attribute their
success, as all else, to divine benevolence" — but it has held up, and Llama,
Mistral, Qwen, and PaLM all use it.

Oracle: the same expression written with torch.nn.functional.silu
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def swiglu_hidden_dim(d_model: int, hidden_mult: float = 4.0, multiple_of: int = 64) -> int:
    """Hidden width for a SwiGLU that costs the same as a plain ``hidden_mult`` MLP.

    SwiGLU has three matrices where a plain MLP has two. To spend the same
    parameters, its hidden layer is two-thirds as wide: ``3 · d · (2/3 · m · d)
    = 2 · d · (m · d)``. So ``hidden_mult=4`` in a config means "the budget of
    a 4x MLP", and the actual hidden width is ``8/3 · d_model``, rounded up to
    ``multiple_of`` so the matrices tile cleanly onto GPU kernels.
    """
    hidden = int(2 * hidden_mult * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """Gated feed-forward layer, applied to each position independently.

    Args:
        d_model: width in and out.
        hidden_mult: the width multiplier of the plain MLP this should cost
            the same as. See :func:`swiglu_hidden_dim`.
        hidden_dim: set the hidden width directly, overriding ``hidden_mult``.
        multiple_of: round the derived hidden width up to a multiple of this.
        bias: add biases to the three projections. Off, as in most modern
            decoders.
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: float = 4.0,
        *,
        hidden_dim: int | None = None,
        multiple_of: int = 64,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = swiglu_hidden_dim(d_model, hidden_mult, multiple_of)
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        for proj in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)  # same scale as the backbone
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        z = self.gate_proj(x)  # [..., hidden]
        gate = z * torch.sigmoid(z)  # SiLU, written out: ~0 when z << 0, ~z when z >> 0
        content = self.up_proj(x)  # [..., hidden]
        return self.down_proj(gate * content)  # [..., d_model]

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"
