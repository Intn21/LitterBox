"""The core interface. Everything in the repo hangs off this file.

A token mixer is one layer's sequence-mixing operation — the thing that moves
information between positions. The MLP/MoE lives outside it. Full attention,
sliding window, MLA, DeltaNet, and DSA are all the same shape behind this
interface, which is what makes a hybrid model a config file rather than a fork.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.infer.cache import KVCache


@dataclass
class MixerState:
    """Union of all inference-time state a mixer might carry.

    Deliberately a union rather than a per-mixer type: generation code and
    profiling can then treat every mixer uniformly. Full attention uses ``kv``;
    the DeltaNet family uses ``recurrent`` (plus ``conv`` for its short
    convolution); sparse mixers such as DSA use ``kv`` together with
    ``indices``.
    """

    kv: KVCache | None = None
    recurrent: Tensor | None = None
    conv: Tensor | None = None
    indices: Tensor | None = None


class TokenMixer(nn.Module, ABC):
    """One layer's sequence-mixing operation."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        """Mix information across positions.

        Args:
            x: ``[batch, seq, d_model]``.
            state: carried inference state, or ``None`` during training, where
                the full sequence is processed at once.
            pos_offset: absolute position of ``x[:, 0]`` in the full sequence.
                Needed so RoPE stays correct during incremental decode.

        Returns:
            The mixed activations ``[batch, seq, d_model]`` and the updated
            state (``None`` when no state was passed in).

        Implementations must satisfy two properties, both enforced by
        ``tests/``: outputs at position ``t`` may not depend on inputs at
        positions ``> t`` (causality), and running the full sequence in one
        parallel call must equal stepping through it one token at a time with
        state (parallel/recurrent consistency).
        """

    @property
    @abstractmethod
    def state_bytes_per_token(self) -> float:
        """Bytes of inference state added per token of context.

        Zero for constant-state mixers such as the DeltaNet family, whose state
        is a fixed-size matrix. This is what lets profiling compare a KV cache
        against a recurrent state on the same axis.
        """
