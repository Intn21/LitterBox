"""Inference state containers. ``KVCache`` today; recurrent and sparse-index
state join it with the mixers that need them.

**Why a cache exists at all.** Attention is causal: token 100's output depends
on the keys and values of tokens 0-99, and those never change once computed —
nothing later in the sequence can reach back and alter them. So when generating
token 101, recomputing the keys and values of tokens 0-100 produces exactly the
numbers it produced a step ago. A KV cache keeps them. Each step then computes
one query, one key and one value — for the new token only — and attends over
what was kept. Generation goes from quadratic work per token to linear.

Only keys and values are cached. A query is used once, by the token that asked
it, and never again.

**Why preallocated.** The obvious cache is a list you ``torch.cat`` onto each
step. That copies the entire history every token, which puts back the quadratic
cost the cache was meant to remove, just in memory traffic instead of matmuls.
This one allocates ``[batch, kv_heads, max_len, head_dim]`` once, writes each new
token into the next free slot, and hands back a *view* of the filled part. No
step copies anything it has already stored.

Each container reports its own byte cost, so profiling can put a growing KV
cache and a constant-size recurrent state on the same axis.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    """Keys and values for one attention layer, written once and read every step.

    Args:
        batch_size: sequences generated together.
        kv_heads: key/value heads — *not* query heads. This is where
            grouped-query attention's saving is actually realised.
        max_len: slots to allocate. Writing past it raises; a full-attention
            cache has nowhere to put token ``max_len``, and silently dropping
            the oldest token would change what every later token attends to.
        head_dim: channels per head.

    Keys are stored **already rotated**. RoPE turns a key by its absolute
    position, which is fixed the moment the token exists, so the rotation is
    done once on the way in and never again.
    """

    def __init__(
        self,
        batch_size: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        shape = (batch_size, kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0  # how many slots hold real tokens; also the next position

    @property
    def max_len(self) -> int:
        return self.k.shape[2]

    @property
    def total(self) -> int:
        """Tokens seen so far — which is also the next token's absolute position."""
        return self.length

    @property
    def first_position(self) -> int:
        """Absolute position of the first key ``append`` returns. Always 0: nothing is dropped."""
        return 0

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store ``[batch, kv_heads, new, head_dim]`` and return everything so far.

        The returned tensors are views of the cache's storage, ``length`` long
        along the sequence axis — the past plus what was just written.
        """
        if k.shape != v.shape:
            raise ValueError(f"key and value shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        expected = (self.k.shape[0], self.k.shape[1], self.k.shape[3])
        if (k.shape[0], k.shape[1], k.shape[3]) != expected:
            raise ValueError(
                f"cache holds [batch, kv_heads, _, head_dim] = "
                f"[{expected[0]}, {expected[1]}, _, {expected[2]}], got {tuple(k.shape)}. "
                f"Keys are cached per KV head, before they are repeated for grouped queries."
            )
        new = k.shape[2]
        end = self.length + new
        if end > self.max_len:
            raise ValueError(
                f"cache is full: {self.length} stored + {new} new > max_len={self.max_len}"
            )
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self) -> None:
        """Forget everything, keep the allocation. For reusing a cache across prompts."""
        self.length = 0

    @property
    def bytes_per_token(self) -> int:
        """What one more token of context costs, per sequence: a key and a value per KV head."""
        _, kv_heads, _, head_dim = self.k.shape
        return 2 * kv_heads * head_dim * self.k.element_size()

    @property
    def nbytes_used(self) -> int:
        """Bytes holding real tokens — grows with context, unlike the allocation."""
        return self.k.shape[0] * self.length * self.bytes_per_token

    @property
    def nbytes_allocated(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    def __repr__(self) -> str:
        b, h, n, d = self.k.shape
        return (
            f"<KVCache {self.length}/{n} tokens, batch={b}, kv_heads={h}, head_dim={d}, "
            f"{self.k.dtype}, {self.nbytes_used / 1e6:.2f} MB used of "
            f"{self.nbytes_allocated / 1e6:.2f} MB>"
        )


class RollingKVCache:
    """Keys and values for the last ``window`` tokens only — a cache that forgets on purpose.

    This is what sliding-window attention buys. A full cache grows by one slot
    per token forever; this one fills up to ``window`` slots and then stays that
    size however long the context gets, because a windowed layer can never look
    further back than that anyway.

    ``append`` returns the kept past followed by the new tokens, and
    :attr:`first_position` says which absolute position the first of those keys
    sits at. The mixer needs that: once old tokens have been dropped, slot ``j``
    no longer holds position ``j``, so the mask has to be built from positions
    rather than from slot indices.

    Written for clarity rather than speed: each append concatenates and keeps the
    tail. That copies at most ``window`` tokens — a fixed cost, not the unbounded
    one that makes ``torch.cat`` the wrong way to build a *full* cache. A ring
    buffer would avoid even that, at the price of keys stored out of order.
    """

    def __init__(
        self,
        batch_size: int,
        kv_heads: int,
        window: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.window = window
        self.k = torch.zeros((batch_size, kv_heads, 0, head_dim), dtype=dtype, device=device)
        self.v = torch.zeros((batch_size, kv_heads, 0, head_dim), dtype=dtype, device=device)
        self.total = 0  # tokens ever seen: the next token's absolute position
        self.first_position = 0  # absolute position of the first key last handed back

    @property
    def length(self) -> int:
        """Tokens currently held — never more than ``window``."""
        return self.k.shape[2]

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Add ``[batch, kv_heads, new, head_dim]``; return the kept past plus the new tokens."""
        if k.shape != v.shape:
            raise ValueError(f"key and value shapes differ: {tuple(k.shape)} vs {tuple(v.shape)}")
        expected = (self.k.shape[0], self.k.shape[1], self.k.shape[3])
        if (k.shape[0], k.shape[1], k.shape[3]) != expected:
            raise ValueError(
                f"cache holds [batch, kv_heads, _, head_dim] = "
                f"[{expected[0]}, {expected[1]}, _, {expected[2]}], got {tuple(k.shape)}. "
                f"Keys are cached per KV head, before they are repeated for grouped queries."
            )
        self.first_position = self.total - self.length
        k_all = torch.cat((self.k, k), dim=2)
        v_all = torch.cat((self.v, v), dim=2)
        self.total += k.shape[2]
        # Keep only what a future query could still reach. The mixer's mask hides
        # the one kept key that sits exactly `window` behind the next query.
        self.k, self.v = k_all[:, :, -self.window :], v_all[:, :, -self.window :]
        return k_all, v_all

    def reset(self) -> None:
        self.k, self.v = self.k[:, :, :0], self.v[:, :, :0]
        self.total = self.first_position = 0

    @property
    def bytes_per_token(self) -> int:
        _, kv_heads, _, head_dim = self.k.shape
        return 2 * kv_heads * head_dim * self.k.element_size()

    @property
    def nbytes_used(self) -> int:
        return self.k.shape[0] * self.length * self.bytes_per_token

    def __repr__(self) -> str:
        b, h, _, d = self.k.shape
        return (
            f"<RollingKVCache holding {self.length} of the last {self.window} tokens "
            f"({self.total} seen), batch={b}, kv_heads={h}, head_dim={d}, {self.k.dtype}, "
            f"{self.nbytes_used / 1e6:.2f} MB>"
        )
