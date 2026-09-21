"""Learned absolute positions: one trainable vector per slot, added once.

GPT-2's scheme, and the simplest positional encoding there is. The table has a
row for every position the model will ever see; row ``p`` is added to whatever
token lands in slot ``p``; backprop tunes the rows exactly the way it tunes the
token embedding. Nothing about order is built in — the model discovers what
"slot 17" should look like, including that it sits next to slot 18.

Two consequences follow from "nothing built in", and they are the reason the
other strategies exist:

- **No row, no position.** A position past ``max_seq_len`` has never been
  trained and has no entry. There is nothing to extrapolate from, so reading
  past the table raises.
- **Neighbours start unrelated.** Rows begin as independent noise. Sinusoidal
  rows for nearby positions start similar; these have to learn it.

This is distinct from ``Sinusoidal(learnable=True)``, which starts from the
sin/cos table — unit-scale entries, order already encoded — and lets training
move it. Here the rows start as small noise at the same scale as the token
embedding (``N(0, 0.02)``), so at step zero neither half of the sum drowns out
the other. That balance is the trap ``demo/positional/sinusoidal.ipynb``
section 6 is about.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from litterbox.positional.base import PositionalEncoding, register_positional


@register_positional("learned")
class Learned(PositionalEncoding):
    """A trainable ``[max_seq_len, d_model]`` table, random at init.

    Args:
        d_model: channel count of the residual stream. Unlike the sinusoidal
            table this need not be even — no channels are paired.
        max_seq_len: number of rows. Also the hard ceiling on sequence length.
        init_std: standard deviation of the initial rows. The default matches
            the backbone's token-embedding init so the two summands start at a
            comparable scale.
    """

    def __init__(self, d_model: int, max_seq_len: int, *, init_std: float = 0.02) -> None:
        super().__init__()
        # A bare Parameter rather than nn.Embedding: positions in a forward
        # pass are always a contiguous run, so the lookup is a slice. The
        # gradient is the same scatter — rows outside the slice get exactly 0.
        self.table = nn.Parameter(torch.empty(max_seq_len, d_model))
        nn.init.normal_(self.table, mean=0.0, std=init_std)

    def embed(self, x: Tensor, pos_offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        if pos_offset + seq_len > self.table.shape[0]:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + seq_len} exceed the "
                f"table's max_seq_len={self.table.shape[0]}; a learned table has "
                f"no row for a position it was never trained on"
            )
        return x + self.table[pos_offset : pos_offset + seq_len].to(x.dtype)
