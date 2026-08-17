"""Causal attention restricted to a fixed-width local window.

The easiest non-trivial mixer, and the one that produces the repo's first
end-to-end signature: SWA-only fails needle retrieval past its window, a
SWA/full hybrid recovers it.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/sliding_window.py`` and must match this one
numerically.

Oracle: full_attention with an explicitly banded mask
Design note: docs/design-notes/sliding_window.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("sliding_window")
class SlidingWindowAttention(TokenMixer):
    """Local causal attention over the last ``window`` positions."""

    def __init__(self, d_model: int, heads: int, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Milestone 2")

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        raise NotImplementedError("Milestone 2")

    @property
    def state_bytes_per_token(self) -> float:
        raise NotImplementedError("Milestone 2")
