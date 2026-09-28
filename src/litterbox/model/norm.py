"""Normalization layers. RMSNorm, and the LayerNorm it replaced.

Deep residual networks have a volume problem. Every block *adds* its output to
the stream, so the stream's magnitude drifts as depth grows, and each layer
would have to cope with inputs at whatever scale the layers before it happened
to produce. A norm in front of each sub-layer fixes the scale of what that
sub-layer reads, so every layer sees inputs of the same size no matter how deep
it sits or how training has moved the weights upstream.

RMSNorm does the least that achieves this: divide each token's vector by its
own root-mean-square, then multiply by a learned per-channel gain.

    rms(x) = sqrt(mean(x_i ** 2) + eps)
    y      = x / rms(x) * gain

LayerNorm, the older choice, also subtracts the mean first and adds a learned
bias after. Zhang & Sennrich (2019) found the re-centering contributes little
and the re-scaling does the work; dropping it is cheaper and is what Llama,
Mistral, Qwen, and most current decoders use.

LayerNorm is kept as the second implementation, and as the one GPT-2 and
nanoGPT use:

    mean(x) = mean(x_i)
    var(x)  = mean((x_i - mean(x)) ** 2)
    y       = (x - mean(x)) / sqrt(var(x) + eps) * gain + bias

Worked through on one token, x = [1, 2, 3, 6]: RMSNorm divides by
sqrt(12.5) = 3.54 and gives [0.28, 0.57, 0.85, 1.70]. LayerNorm first subtracts
the mean, 3, then divides by sqrt(3.5) = 1.87 and gives [-1.07, -0.53, 0, 1.60].
Add 10 to every entry and LayerNorm's answer does not move; RMSNorm's does.

Oracles: torch.nn.RMSNorm, torch.nn.LayerNorm
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the channel axis.

    Args:
        d_model: width of the last axis; one learned gain per channel.
        eps: added under the square root so an all-zero vector (a padding
            row, a dead token) gives zeros rather than NaN. It is also a
            floor: once activations shrink to around ``sqrt(eps)`` — about
            3e-3 at the default — eps dominates ``mean(x**2)`` and the output
            is no longer unit scale. So it encodes the smallest activations
            you expect, not only a guard against dividing by zero.

    The norm is per token: each vector is scaled by its own RMS and nothing
    else, so no information moves between positions and causality is
    untouched.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Ones, not noise: at init the layer is a pure normalizer, and the
        # model learns per channel how much of that to undo.
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # Squares of bf16 activations overflow and underflow easily, and this
        # reduction feeds every sub-layer in the model. Do it in fp32.
        x32 = x.float()
        mean_square = x32.pow(2).mean(dim=-1, keepdim=True)  # [..., 1]
        normed = x32 * torch.rsqrt(mean_square + self.eps)  # [..., d]: unit RMS
        return normed.to(x.dtype) * self.gain

    def extra_repr(self) -> str:
        return f"{self.d_model}, eps={self.eps}"


class LayerNorm(nn.Module):
    """Mean-and-variance normalization over the channel axis.

    Args:
        d_model: width of the last axis; one learned gain (and bias) per channel.
        eps: added to the variance under the square root, with the same floor
            as :class:`RMSNorm`'s.
        bias: learn a per-channel bias added after the gain. On, as in the
            original and in ``torch.nn.LayerNorm``.

    Per token, like :class:`RMSNorm`: the mean and variance are each vector's
    own, so nothing moves between positions.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, *, bias: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()  # the same fp32 reduction as RMSNorm, for the same reason
        centered = x32 - x32.mean(dim=-1, keepdim=True)  # [..., d]: zero mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)  # [..., 1]
        normed = centered * torch.rsqrt(variance + self.eps)  # [..., d]: unit variance
        y = normed.to(x.dtype) * self.gain
        return y if self.bias is None else y + self.bias

    def extra_repr(self) -> str:
        return f"{self.d_model}, eps={self.eps}, bias={self.bias is not None}"
