"""Sinusoidal absolute positions, added to the residual stream once.

The original Transformer's scheme: position ``p``, channel pair ``i`` gets
``sin(p / base^(2i/d))`` and ``cos(p / base^(2i/d))``. Geometrically each pair
of channels is a clock hand, and the hands sweep at frequencies from one radian
per token (pair 0) down to one radian per ``base`` tokens (the last pair) — so
any position within the wavelength range has a unique fingerprint, and nearby
positions have similar ones.

``learnable=True`` keeps the same table but registers it as a ``Parameter``,
so training can move it. Starting from the sinusoid rather than noise means
the learnable variant begins with a sensible notion of order instead of having
to discover one — and whatever it drifts toward is measurable against where it
started.

Not to be confused with :class:`~litterbox.positional.learned.Learned`, GPT-2's
scheme, whose table starts as small random noise with no order built in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from litterbox.positional.base import PositionalEncoding, register_positional


def sinusoidal_table(max_seq_len: int, d_model: int, base: float = 10000.0) -> Tensor:
    """The ``[max_seq_len, d_model]`` sin/cos table, sin on even channels."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even to pair channels, got {d_model}")
    pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    inv_freq = base ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # [d/2]
    angles = pos * inv_freq  # [L, d/2]
    table = torch.empty(max_seq_len, d_model)
    table[:, 0::2] = angles.sin()
    table[:, 1::2] = angles.cos()
    return table


@register_positional("sinusoidal")
class Sinusoidal(PositionalEncoding):
    """Additive sin/cos positions; optionally a trainable table.

    Args:
        d_model: channel count of the residual stream.
        max_seq_len: longest position the table covers. Reading past it
            raises rather than wrapping or extrapolating.
        base: the wavelength knob; 10000 is the paper's choice.
        learnable: register the table as a ``Parameter`` (initialized to the
            sinusoid) instead of a fixed buffer.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        base: float = 10000.0,
        *,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        table = sinusoidal_table(max_seq_len, d_model, base)
        if learnable:
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table, persistent=False)

    def embed(self, x: Tensor, pos_offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        if pos_offset + seq_len > self.table.shape[0]:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + seq_len} exceed the "
                f"table's max_seq_len={self.table.shape[0]}"
            )
        return x + self.table[pos_offset : pos_offset + seq_len].to(x.dtype)
