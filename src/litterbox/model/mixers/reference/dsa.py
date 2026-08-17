"""DSA-style learned sparsity: a lightning indexer plus top-k selection.

Wraps full attention rather than replacing it: a few ReLU indexer heads
score cached tokens, the top-k are selected, and dense attention runs over
just those. Trained in two stages — freeze the model and fit the indexer
with a KL loss against the dense attention distribution, then train jointly.

The interleaved/non-interleaved RoPE mismatch in the indexer is a
documented exercise, not an accident; see the design note.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. A fused or
chunked version belongs in ``mixers/fast/dsa.py`` and must match this one
numerically.

Oracle: full_attention with k = full context (selection disabled)
Design note: docs/design-notes/dsa.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("dsa")
class SparseIndexedAttention(TokenMixer):
    """Top-k sparse attention driven by a learned lightning indexer."""

    def __init__(self, d_model: int, heads: int, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("Milestone 4")

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        raise NotImplementedError("Milestone 4")

    @property
    def state_bytes_per_token(self) -> float:
        raise NotImplementedError("Milestone 4")
