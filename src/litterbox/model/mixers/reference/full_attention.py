"""Full causal attention with grouped-query heads and an injected positional strategy.

The correctness anchor for the whole repo: every other mixer is compared
against this one on short sequences where they should agree.

Each token asks a question (its query), every earlier token offers a label
(its key) and a payload (its value). The token scores its query against every
key, turns the scores into weights that sum to one, and takes the weighted
average of the payloads. "Causal" means a token may only look at itself and
what came before it. "Multi-head" means the model runs several of these in
parallel on narrower vectors, so one head can track syntax while another
tracks coreference. "Grouped-query" means several query heads share one
key/value head, which shrinks the KV cache without shrinking the number of
questions being asked.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation is the test oracle and the pedagogical product. The
attention math is written out rather than delegated to
``F.scaled_dot_product_attention``, which is this file's *oracle*, not its
implementation. A fused or chunked version belongs in
``mixers/fast/full_attention.py`` and must match this one numerically.

Oracle: torch.nn.functional.scaled_dot_product_attention
Design note: docs/design-notes/full-attention.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from litterbox.model.mixers.base import MixerState, TokenMixer
from litterbox.model.registry import register_mixer
from litterbox.positional import NoPE, PositionalEncoding

if TYPE_CHECKING:
    from torch import Tensor


@register_mixer("full_attention")
class FullAttention(TokenMixer):
    """Multi-head / grouped-query causal attention.

    Args:
        d_model: width of the residual stream, in and out.
        heads: number of query heads. ``d_model`` must divide evenly; each
            head works on ``head_dim = d_model // heads`` channels.
        kv_heads: number of key/value heads. Defaults to ``heads`` (plain
            multi-head attention). ``1`` is multi-query attention; anything in
            between is grouped-query, with ``heads // kv_heads`` query heads
            sharing each key/value head.
        pos: the positional strategy, injected as an instance. Its ``rotate``
            hook is called unconditionally on the queries and keys — rotary
            strategies act there, additive ones and NoPE pass through. The
            mixer never inspects what it was given, so a NoPE layer in a
            hybrid is a config edit, not a code path. A rotary strategy must
            be built for this mixer's ``head_dim``; RoPE raises otherwise.
        bias: add biases to the four projections. Off, as in most modern
            decoders.

    No ``**kwargs``: a misspelled config key should raise here rather than be
    swallowed.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_heads: int | None = None,
        pos: PositionalEncoding | None = None,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        kv_heads = heads if kv_heads is None else kv_heads
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
        if heads % kv_heads != 0:
            raise ValueError(
                f"heads={heads} must be divisible by kv_heads={kv_heads}: each key/value "
                f"head serves a whole number of query heads"
            )
        self.d_model = d_model
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // heads
        self.group_size = heads // kv_heads

        # Queries get one projection per head; keys and values only per KV
        # head. That asymmetry is the entire parameter and cache saving of GQA.
        self.q_proj = nn.Linear(d_model, heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * self.head_dim, d_model, bias=bias)
        # The last projection before the residual add: opt in to depth-scaled init
        # (see model.block.scale_residual_projections).
        self.o_proj.residual_out = True
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)  # same scale as the backbone
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        self.pos = pos if pos is not None else NoPE()

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

        # 1. Project, then split the channel axis into heads. The view must
        #    come before the transpose: channels are laid out head-by-head, so
        #    [b, s, h*d] -> [b, s, h, d] is a pure reshape, and only then do
        #    heads move next to batch so each head attends independently.
        q = self.q_proj(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)  # [b, h,  s, d]
        k = self.k_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)  # [b, kv, s, d]
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)  # [b, kv, s, d]

        # 2. Position. Queries and keys only — values are never scored against
        #    anything, so turning them would leak absolute position into the
        #    output. Done before the KV heads are repeated: rotating kv heads
        #    once is cheaper than rotating h copies, and it is what gets cached.
        q, k = self.pos.rotate(q, k, pos_offset)

        # 3. Grouped-query: hand each KV head to its group of query heads.
        #    repeat_interleave gives [kv0, kv0, kv1, kv1, ...], so query heads
        #    0..g-1 share KV head 0. That grouping is a convention weights are
        #    trained against, and it matches SDPA's enable_gqa.
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)  # [b, h, s, d]
            v = v.repeat_interleave(self.group_size, dim=1)  # [b, h, s, d]

        # 4. Score every query against every key. Scaled by sqrt(head_dim) —
        #    the width of the vectors being dotted, not d_model — so the scores
        #    have unit variance at init and softmax does not start saturated.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # [b, h, s, s]

        # 5. Causal mask: query i may see key j only when j <= i. Masked
        #    scores become -inf, which softmax turns into exactly 0 — not a
        #    small number, zero, which is what makes the causality test able
        #    to demand bit-identical outputs.
        allowed = torch.ones(s, s, dtype=torch.bool, device=x.device).tril()  # [s, s]
        scores = scores.masked_fill(~allowed, float("-inf"))

        # 6. Scores -> weights. Softmax in fp32 whatever the activation dtype:
        #    exp() of a bf16 score overflows and underflows far too easily.
        weights = scores.float().softmax(dim=-1).to(q.dtype)  # [b, h, s, s]

        # 7. Weighted average of the payloads, then undo the head split —
        #    transpose back first, so heads are adjacent to channels again
        #    before they are flattened together.
        out = weights @ v  # [b, h, s, d]
        out = out.transpose(1, 2).reshape(b, s, self.heads * self.head_dim)  # [b, s, h*d]
        return self.o_proj(out), None

    @property
    def state_bytes_per_token(self) -> float:
        """One key and one value vector per KV head, per token of context.

        This is what grouped-query attention buys: the cache scales with
        ``kv_heads``, not ``heads``.
        """
        return float(2 * self.kv_heads * self.head_dim * self.k_proj.weight.element_size())
