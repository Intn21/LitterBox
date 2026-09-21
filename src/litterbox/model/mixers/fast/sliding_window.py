"""Sliding-window attention through PyTorch's fused kernel.

The same mixin as the reference, over the fast forward pass: the band comes from
``_allowed()`` and is handed to ``scaled_dot_product_attention`` as an explicit
mask. That rules out the FlashAttention backend, which only knows plain causal
masks, so this runs on the memory-efficient kernel instead — still without
materialising the score matrix for the backward pass. A kernel that *skips* the
masked blocks, and so makes training cost scale with ``window`` rather than with
sequence length, is further work.

Oracle: litterbox.model.mixers.reference.sliding_window.SlidingWindowAttention
"""

from __future__ import annotations

from litterbox.model.mixers.fast.full_attention import FastFullAttention
from litterbox.model.mixers.reference.sliding_window import Windowed
from litterbox.model.registry import register_mixer
from litterbox.positional import PositionalEncoding


@register_mixer("sliding_window_fast")
class FastSlidingWindowAttention(Windowed, FastFullAttention):
    """:class:`SlidingWindowAttention` with scoring, masking and softmax fused."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_heads: int | None = None,
        pos: PositionalEncoding | None = None,
        *,
        window: int,
        bias: bool = False,
    ) -> None:
        super().__init__(d_model, heads, kv_heads, pos, bias=bias)
        self._init_window(window)
