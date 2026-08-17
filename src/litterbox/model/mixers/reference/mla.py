"""Multi-head latent attention: low-rank KV compression with decoupled RoPE.

Much easier to get right once ``positional/rope.py`` makes layouts explicit,
since the decoupled path is precisely where layout mistakes hide.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/mla.py`` and must match this one
numerically.

Oracle: full_attention at full rank (compression disabled)
Design note: docs/design-notes/mla.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("mla")
class MultiHeadLatentAttention(TokenMixer):
    """Attention over a compressed latent KV representation."""

    def __init__(self, d_model: int, heads: int, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Milestone 5")

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        raise NotImplementedError("Milestone 5")

    @property
    def state_bytes_per_token(self) -> float:
        raise NotImplementedError("Milestone 5")
