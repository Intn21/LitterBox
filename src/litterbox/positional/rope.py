"""Rotary position embeddings, with layout as an explicit, named parameter.

RoPE encodes position by *rotating* each query and key head vector, treating
its channels as 2D pairs: the pair for frequency ``i`` at position ``p`` is
rotated by angle ``p / base^(2i/head_dim)``. Because rotations compose by
adding angles, the dot product between a query at position ``m`` and a key at
position ``n`` depends only on ``m - n`` — absolute position cancels, relative
position remains. That shift invariance is the entire point, and it is what
``tests/test_positional.py`` asserts directly.

The two layouts disagree about *which channels form a pair*:

- ``interleaved`` (GPT-J, GPT-NeoX weights on disk): adjacent channels —
  ``(0,1), (2,3), ...``
- ``half`` (Llama, and most HF ``rotate_half`` code): split at the middle —
  ``(0, d/2), (1, d/2+1), ...``

Either is correct alone; both preserve norms and shift invariance. But a query
rotated in one layout against a key rotated in the other pairs the wrong
channels, and nothing crashes — quality just silently degrades. That is the
bug class that bit DeepSeek's DSA indexer, why ``layout`` is a required
constructor argument here rather than a default, and why the mismatch has its
own test.
"""

from __future__ import annotations

import torch
from torch import Tensor

from litterbox.positional.base import PositionalEncoding, register_positional

LAYOUTS = ("interleaved", "half")


@register_positional("rope")
class RoPE(PositionalEncoding):
    """Rotate queries and keys; the residual stream is never touched.

    Args:
        head_dim: channels per attention head — RoPE acts per head, so this is
            ``d_model // n_heads``, not ``d_model``. Must be even.
        max_seq_len: longest position with precomputed angles. Reading past it
            raises rather than silently recomputing, so length-extension is an
            explicit decision (YaRN's job) and never an accident.
        layout: ``"interleaved"`` or ``"half"``. No default on purpose — the
            caller must say which convention its weights use.
        base: rotation wavelength knob; 10000 unless extending context.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        *,
        layout: str,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even to form rotation pairs, got {head_dim}")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
        self.head_dim = head_dim
        self.layout = layout

        # Angles for every (position, frequency) pair, fp32: [max_seq_len, head_dim/2].
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = pos * inv_freq
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def rotate(self, q: Tensor, k: Tensor, pos_offset: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[-2]
        if pos_offset + seq_len > self.cos.shape[0]:
            raise ValueError(
                f"positions {pos_offset}..{pos_offset + seq_len} exceed the "
                f"precomputed max_seq_len={self.cos.shape[0]}; extending context "
                f"is yarn.py's job, not an implicit recompute"
            )
        cos = self.cos[pos_offset : pos_offset + seq_len]  # [S, head_dim/2]
        sin = self.sin[pos_offset : pos_offset + seq_len]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"expected head_dim={self.head_dim} in the last axis, got {x.shape[-1]} "
                f"— RoPE applies per head, after splitting d_model into heads"
            )
        # Rotation happens in fp32 regardless of activation dtype: sin/cos of
        # large angles lose too much in bf16, and this runs once per layer.
        x1, x2 = self._split(x.float())
        out = self._join(x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        return out.to(x.dtype)

    # The layout IS the split/join choice; the rotation math above is shared.

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.layout == "interleaved":
            return x[..., 0::2], x[..., 1::2]
        return x.chunk(2, dim=-1)

    def _join(self, a: Tensor, b: Tensor) -> Tensor:
        if self.layout == "interleaved":
            return torch.stack((a, b), dim=-1).flatten(-2)
        return torch.cat((a, b), dim=-1)
