"""Full causal attention with grouped-query heads and RoPE.

The correctness anchor for the whole repo: every other mixer is compared
against this one on short sequences where they should agree.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/full_attention.py`` and must match this one
numerically.

Oracle: torch.nn.functional.scaled_dot_product_attention
Design note: docs/design-notes/full_attention.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("full_attention")
class FullAttention(TokenMixer):
    """Multi-head / grouped-query causal attention."""

    def __init__(self, d_model: int, heads: int, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Milestone 1: the first mixer to implement.")

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        raise NotImplementedError("Milestone 1: the first mixer to implement.")

    @property
    def state_bytes_per_token(self) -> float:
        raise NotImplementedError("Milestone 1: the first mixer to implement.")
