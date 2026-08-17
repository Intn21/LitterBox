"""Gated DeltaNet: DeltaNet plus a data-dependent decay gate on the state.

The workhorse of the flagship ablation — full-attention control vs pure GDN
vs hybrids at 7:1, 3:1, and 1:1, swept across context lengths.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/gated_deltanet.py`` and must match this one
numerically.

Oracle: flash-linear-attention (test dependency only)
Design note: docs/design-notes/gated_deltanet.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("gated_deltanet")
class GatedDeltaNet(TokenMixer):
    """Delta-rule linear attention with learned state decay."""

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
