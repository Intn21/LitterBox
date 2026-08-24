"""Rotary position embeddings, with layout as an explicit, named parameter.

RoPE gives a token its sense of position by *rotating* its query and key vectors
by an angle proportional to where the token sits. The dot product between two
rotated vectors then depends on how far apart they are, which is exactly the
signal attention needs.

The rotation acts on *pairs* of channels, and there are two conventions for which
channels pair up. For a head vector ``[a0 a1 a2 a3 a4 a5 a6 a7]``:

    interleaved  : (a0,a1) (a2,a3) (a4,a5) (a6,a7)   — adjacent, GPT-J / the paper
    half         : (a0,a4) (a1,a5) (a2,a6) (a3,a7)   — split in half, HF Llama

Both are correct in isolation. Mixing them produces no error and no crash — the
model simply gets quietly worse, which is what makes it such an expensive bug.
It is also the bug that bit DeepSeek's DSA indexer, so both layouts are
implemented, named, and tested here rather than left implicit.

``half`` is the default because HuggingFace's Llama uses it, which means our RoPE
can be checked against a real model's output.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

RopeLayout = Literal["half", "interleaved"]


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    *,
    base: float = 10000.0,
    layout: RopeLayout = "half",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute the cos/sin tables for every position up to ``max_seq_len``.

    The angles never change during training, so they are built once and sliced
    per forward pass.

    Returns:
        ``(cos, sin)``, each ``[max_seq_len, head_dim]`` — already expanded to
        full head width in the arrangement ``layout`` expects, so the caller
        never has to think about pairing again.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    # One frequency per *pair* of channels: high frequencies rotate fast (local
    # detail), low frequencies rotate slowly (long-range position).
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (base**exponent)  # [head_dim/2]

    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # [max_seq_len]
    freqs = torch.outer(pos, inv_freq)  # [max_seq_len, head_dim/2]

    if layout == "half":
        # Channel i pairs with channel i + head_dim/2, so the table is the
        # half-width angles concatenated with themselves.
        angles = torch.cat((freqs, freqs), dim=-1)
    elif layout == "interleaved":
        # Channel 2i pairs with 2i+1, so each angle is repeated side by side.
        angles = freqs.repeat_interleave(2, dim=-1)
    else:
        raise ValueError(f"unknown rope layout {layout!r}; expected 'half' or 'interleaved'")

    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Pair channel i with channel i + D/2. Matches HuggingFace Llama."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x: Tensor) -> Tensor:
    """Pair channel 2i with channel 2i+1. Matches GPT-J and the original paper."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


_ROTATE = {"half": _rotate_half, "interleaved": _rotate_interleaved}


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    layout: RopeLayout = "half",
) -> tuple[Tensor, Tensor]:
    """Rotate ``q`` and ``k`` in place of their positions.

    Args:
        q: ``[B, H, S, Dh]``.
        k: ``[B, H_kv, S, Dh]`` — may have fewer heads than ``q`` under GQA.
        cos, sin: ``[S, Dh]``, already sliced to this call's positions by the
            caller. Slicing is the caller's job because only the mixer knows its
            ``pos_offset`` during decode.
        layout: which channels pair up. Must match whatever produced ``cos``/``sin``.

    Returns:
        The rotated ``q`` and ``k``, same shapes as the inputs.
    """
    rotate = _ROTATE.get(layout)
    if rotate is None:
        raise ValueError(f"unknown rope layout {layout!r}; expected 'half' or 'interleaved'")

    # [S, Dh] -> [1, 1, S, Dh] so it broadcasts over batch and heads alike.
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)

    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
