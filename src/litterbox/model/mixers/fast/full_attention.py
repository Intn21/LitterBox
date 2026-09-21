"""Full attention through PyTorch's fused kernel — the fast twin of the reference.

Same parameters, same numbers, different cost. The reference mixer builds the
whole ``[batch, heads, seq, seq]`` score matrix, which is what makes it readable
and what makes it unusable at scale: at batch 32 and 2,048 tokens that one
tensor is 6.4 GB per layer, and training keeps several per layer for the
backward pass. ``F.scaled_dot_product_attention`` computes the same result
without ever materialising it. On CUDA it dispatches to FlashAttention, which
works through the matrix in blocks that fit in on-chip memory, so memory grows
linearly with sequence length instead of quadratically.

This class subclasses the reference and overrides only ``forward``. The
projections, their init, the state-dict keys, and ``state_bytes_per_token`` are
inherited, so a checkpoint trained with one tier loads into the other, and the
reference stays the oracle this file is tested against — not the paper.

Oracle: litterbox.model.mixers.reference.full_attention.FullAttention
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F

from litterbox.model.mixers.reference.full_attention import FullAttention
from litterbox.model.registry import register_mixer

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.model.mixers.base import MixerState


@register_mixer("full_attention_fast")
class FastFullAttention(FullAttention):
    """:class:`FullAttention` with scoring, masking, and softmax fused into one call."""

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        if state is not None:
            raise NotImplementedError(
                "Incremental decode needs the KV cache from infer/cache.py (ROADMAP step 3). "
                "The training path, state=None, is complete."
            )
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)  # [b, h,  s, d]
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)  # [b, kv, s, d]
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)  # [b, kv, s, d]
        q, k = self.pos.rotate(q, k, pos_offset)

        # Steps 3-7 of the reference, in one kernel: share KV heads across query
        # groups (enable_gqa uses the same interleaved grouping), score, scale by
        # sqrt(head_dim), mask causally, softmax, and average the values.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.group_size > 1
        )  # [b, h, s, d]

        out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o_proj(out), None
