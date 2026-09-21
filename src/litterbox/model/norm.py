"""Normalization layers. RMSNorm today; a second one joins it in ROADMAP step 5.

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

Oracle: torch.nn.RMSNorm
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
