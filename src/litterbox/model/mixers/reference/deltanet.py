"""DeltaNet: linear attention with the delta rule as its state update.

The delta rule replaces additive accumulation with an error-correcting
write, which is what gives the family its associative-recall behaviour.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/deltanet.py`` and must match this one
numerically.

Oracle: flash-linear-attention (test dependency only)
Design note: docs/design-notes/deltanet.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("deltanet")
class DeltaNet(TokenMixer):
    """Delta-rule linear attention with a short convolution."""

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
