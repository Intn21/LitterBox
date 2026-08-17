"""Vanilla linear attention — kernel feature map, no gating, no delta rule.

Educational baseline: the simplest mixer with a constant-size recurrent
state, and therefore the cheapest place to get the parallel/recurrent
consistency test right before the harder members of the family.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/linear_attention.py`` and must match this one
numerically.

Oracle: its own quadratic form on short sequences
Design note: docs/design-notes/linear_attention.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("linear_attention")
class LinearAttention(TokenMixer):
    """Linear attention with a fixed-size matrix state."""

    def __init__(self, d_model: int, heads: int, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Milestone 3")

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        raise NotImplementedError("Milestone 3")

    @property
    def state_bytes_per_token(self) -> float:
        raise NotImplementedError("Milestone 3")
