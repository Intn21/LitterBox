"""The core interface. Everything in the repo hangs off this file.

A token mixer is the one place in a decoder-only transformer where information
moves *between* positions. Embeddings, channel mixers, norms, and the LM head all
operate on each token independently — so if you want to change how a model
handles long context, this is the only thing to change.

Full attention, sliding window, MLA, DeltaNet, and DSA are all the same shape
behind this interface, which is what makes a hybrid model a config file rather
than a fork.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.infer.cache import KVCache


@dataclass
class MixerState:
    """Union of all inference-time state a mixer might carry.

    Deliberately a union rather than a per-mixer type: generation code can then
    thread state through every layer without knowing what any of them contain.
    Full attention uses ``kv``; the DeltaNet family uses ``recurrent`` (plus
    ``conv`` for its short convolution); sparse mixers such as DSA use ``kv``
    together with ``indices``.
    """

    kv: KVCache | None = None
    recurrent: Tensor | None = None
    conv: Tensor | None = None
    indices: Tensor | None = None


class TokenMixer(nn.Module, ABC):
    """One layer's sequence-mixing operation. ``[B, S, D]`` in, ``[B, S, D]`` out.

    A mixer owns its own projections. It is handed the residual stream and
    returns something the same shape — what happens in between, including how
    many matrices it needs and whether it applies RoPE, is entirely its business.
    That is what keeps :class:`~litterbox.model.block.TransformerBlock` from ever
    growing a conditional about which mixer it holds.
    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        """Mix information across positions.

        Args:
            x: ``[B, S, D]``.
            state: carried inference state, or ``None`` during training, where
                the whole sequence is processed at once.
            pos_offset: absolute position of ``x[:, 0]`` in the full sequence.
                During decode you pass a single token, and without this the mixer
                has no way to know it is the 57th rather than the first.

        Returns:
            The mixed activations ``[B, S, D]`` and the advanced state (``None``
            when no state was passed in).

        There is deliberately **one** entry point rather than a separate ``step``
        method. Mixers whose parallel and recurrent forms differ branch on
        ``x.shape[1]`` internally. Two public methods drift apart, and
        ``tests/test_state_consistency.py`` exists precisely because those two
        paths disagree in practice — behind one method, the test compares the
        same function against itself.

        Implementations must satisfy two properties, both enforced by ``tests/``:
        outputs at position ``t`` may not depend on inputs at positions ``> t``,
        and running a sequence in one call must equal stepping through it one
        token at a time with state.
        """

    def init_state(
        self,
        batch: int,
        max_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> MixerState:
        """Allocate empty inference state for a generation run.

        The mixer allocates, because only the mixer knows the shape: a KV cache
        needs ``[B, H_kv, max_len, Dh]`` while a recurrent state needs a
        fixed-size matrix that ignores ``max_len`` entirely. Generation code
        cannot make that decision, so it does not try.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support stateful decoding yet")

    @property
    @abstractmethod
    def state_bytes_per_token(self) -> float:
        """Bytes of inference state added per token of context.

        Zero for constant-state mixers such as the DeltaNet family, whose state
        is a fixed-size matrix. This is what lets profiling put a growing KV
        cache and a constant recurrent state on the same axis.
        """
