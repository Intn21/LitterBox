"""Inference state containers.

While generating, you produce one token at a time. Recomputing the whole prefix
at every step would be enormous waste, so each mixer carries something forward
between steps. What that something *is* differs by family, and the difference is
the point:

- :class:`KVCache` grows by one entry per token, forever. That is attention.
- A recurrent state is a fixed-size matrix, overwritten each step and never
  growing. That is the linear/DeltaNet family, and constant memory is its whole
  selling point.

Generation code must not care which it is holding, so both live behind
:class:`~litterbox.model.mixers.base.MixerState` and both report their own byte
cost through ``state_bytes_per_token``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Preallocated key/value cache for one attention layer.

    The buffers are allocated once at their full size and filled in place, which
    avoids reallocating a growing tensor on every decode step. ``length`` tracks
    how much of them is real.
    """

    k: Tensor  # [B, H_kv, max_len, Dh]
    v: Tensor  # [B, H_kv, max_len, Dh]
    length: int = 0

    @classmethod
    def empty(
        cls,
        batch: int,
        kv_heads: int,
        max_len: int,
        head_dim: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> KVCache:
        shape = (batch, kv_heads, max_len, head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
            length=0,
        )

    @property
    def max_len(self) -> int:
        return self.k.shape[2]

    def update(self, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor, KVCache]:
        """Append ``k_new``/``v_new`` and return everything cached so far.

        Args:
            k_new, v_new: ``[B, H_kv, S_new, Dh]``.

        Returns:
            The full ``k`` and ``v`` up to the new length, plus the advanced
            cache. The buffers are shared with this instance — only ``length``
            differs — so the return is cheap while still handing back a value
            rather than mutating the caller's object silently.
        """
        start = self.length
        end = start + k_new.shape[2]
        if end > self.max_len:
            raise ValueError(
                f"KV cache overflow: {end} tokens requested but only {self.max_len} allocated. "
                f"Raise max_len when calling init_state()."
            )

        self.k[:, :, start:end] = k_new
        self.v[:, :, start:end] = v_new
        return self.k[:, :, :end], self.v[:, :, :end], replace(self, length=end)
