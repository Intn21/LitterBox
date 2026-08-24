"""Normalization layers.

A norm is a compartment like any other: ``[B, S, D]`` in, ``[B, S, D]`` out, no
state, no communication between tokens. It rescales each token's vector on its
own, which is why swapping one is a config field rather than a code change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from litterbox.model.registry import NORMS


@NORMS.register("rmsnorm")
class RMSNorm(nn.Module):
    """Root-mean-square normalization.

    LayerNorm without the mean subtraction: divide each token's vector by its own
    RMS, then scale by a learned per-channel weight. Cheaper than LayerNorm and
    the standard choice in modern decoder-only models.

    The reduction runs in fp32 even when the model is in bf16. Summing D squared
    values is exactly the kind of accumulation that loses too much in 16 bits, and
    a norm that drifts is very hard to attribute later.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


@NORMS.register("layernorm")
class LayerNorm(nn.Module):
    """Standard LayerNorm, kept so the norm compartment has two implementations.

    A seam with only one thing behind it is a hypothesis, not an abstraction.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        out = torch.nn.functional.layer_norm(
            x.float(), (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps
        )
        return out.to(dtype)


def build_norm(d_model: int, *, type: str = "rmsnorm", eps: float = 1e-5) -> nn.Module:
    """Construct the norm named in config."""
    return NORMS.get(type)(d_model, eps=eps)
