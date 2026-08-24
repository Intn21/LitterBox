"""Full causal attention with grouped-query heads and RoPE.

Reference tier: pure PyTorch, readable, obviously correct. This is the
correctness anchor for the whole repo — every other mixer is compared against it
on short sequences where they should agree.

Shape convention, which is worth stating once and then trusting: the residual
stream is ``[B, S, D]``, and inside this file heads are laid out as
``[B, H, S, Dh]``. That ordering is what ``scaled_dot_product_attention`` takes,
so the oracle comparison is a direct call with no reshaping to get wrong. Note
that ``x.view(B, H, S, Dh)`` would produce the right *shape* from ``[B, S, D]``
while scrambling which channels belong to which head, and PyTorch will not
complain — hence ``.view(B, S, H, Dh).transpose(1, 2)`` everywhere below.

Design note: docs/design-notes/full-attention.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from litterbox.infer.cache import KVCache
from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer
from litterbox.positional.rope import RopeLayout, apply_rope, build_rope_cache

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("full_attention")
class FullAttention(TokenMixer):
    """Multi-head / grouped-query causal attention.

    Grouped-query attention shares each key/value head across several query
    heads. With ``heads=12, kv_heads=2`` the KV cache is six times smaller than
    full multi-head, for very little quality cost — which is why essentially
    every modern model does it.

    Args:
        d_model: model width.
        heads: number of query heads.
        kv_heads: number of key/value heads. Defaults to ``heads`` (plain MHA).
            Must divide ``heads``.
        pos: ``"rope"`` or ``"nope"``. Per-layer, because hybrids deliberately
            drop positional encoding on their full-attention layers.
        rope_layout: ``"half"`` or ``"interleaved"``. See positional/rope.py —
            mixing the two is silent and expensive.
        max_seq_len: how far to precompute the RoPE tables.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        *,
        kv_heads: int | None = None,
        pos: str = "rope",
        rope_layout: RopeLayout = "half",
        rope_base: float = 10000.0,
        max_seq_len: int = 8192,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if d_model % heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by heads {heads}")
        kv_heads = heads if kv_heads is None else kv_heads
        if heads % kv_heads != 0:
            raise ValueError(f"heads {heads} must be divisible by kv_heads {kv_heads}")

        self.d_model = d_model
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // heads
        self.n_rep = heads // kv_heads  # query heads per kv head
        self.pos = pos

        self.q_proj = nn.Linear(d_model, heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * self.head_dim, d_model, bias=bias)

        if pos == "rope":
            self.rope_layout: RopeLayout = rope_layout
            cos, sin = build_rope_cache(
                self.head_dim, max_seq_len, base=rope_base, layout=rope_layout
            )
            # Buffers so they move with .to(device) and stay out of state_dict.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        elif pos != "nope":
            raise ValueError(f"unknown pos {pos!r} for full_attention; expected 'rope' or 'nope'")

    # ------------------------------------------------------------------ state

    def init_state(
        self,
        batch: int,
        max_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MixerState:
        return MixerState(
            kv=KVCache.empty(
                batch, self.kv_heads, max_len, self.head_dim, device=device, dtype=dtype
            )
        )

    @property
    def state_bytes_per_token(self) -> float:
        """Two tensors (k and v), one entry per kv head, per token."""
        return 2 * self.kv_heads * self.head_dim * torch.finfo(torch.bfloat16).bits / 8

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        B, S, _ = x.shape

        # Project, then split into heads. [B, S, D] -> [B, H, S, Dh].
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.pos == "rope":
            # Slice the tables to *this call's* absolute positions. During decode
            # S is 1 and pos_offset says where that single token actually sits.
            cos = self.rope_cos[pos_offset : pos_offset + S]
            sin = self.rope_sin[pos_offset : pos_offset + S]
            if cos.shape[0] < S:
                raise ValueError(
                    f"RoPE table exhausted at position {pos_offset + S}; "
                    f"raise max_seq_len (currently {self.rope_cos.shape[0]})"
                )
            q, k = apply_rope(q, k, cos, sin, layout=self.rope_layout)

        # Fold the cache in. Keys and values now cover the whole prefix; queries
        # only cover the new tokens.
        if state is not None and state.kv is not None:
            k, v, new_kv = state.kv.update(k, v)
            state = MixerState(kv=new_kv)

        # GQA: every kv head serves n_rep query heads.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if state is None:
            # Training / full-sequence: queries and keys align, so the built-in
            # causal flag is correct and fast.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # With a cache, q and k have different lengths and `is_causal=True`
            # would align the mask to the top-left — wrong. Build it explicitly
            # from absolute positions instead.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=self._causal_mask(S, k.shape[2], pos_offset, x.device)
            )

        # [B, H, S, Dh] -> [B, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, self.heads * self.head_dim)
        return self.o_proj(out), state

    @staticmethod
    def _causal_mask(q_len: int, k_len: int, pos_offset: int, device) -> Tensor:
        """Boolean mask, True where a query is allowed to read a key."""
        q_pos = torch.arange(q_len, device=device) + pos_offset
        k_pos = torch.arange(k_len, device=device)
        return k_pos[None, :] <= q_pos[:, None]
