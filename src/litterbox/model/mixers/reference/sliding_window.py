"""Causal attention restricted to a fixed-width local window.

The easiest non-trivial mixer, and the one that produces the repo's first
end-to-end signature: SWA-only fails needle retrieval past its window, a
SWA/full hybrid recovers it.

A token sees itself and the ``window - 1`` tokens before it, and nothing older.
Compared with full attention that is one extra condition in the mask::

    full attention:    key j is visible to the query at p   when  j <= p
    sliding window:    ...                                  when  p - window < j <= p

and that one condition is the whole mixer. Projections, heads, grouped queries,
RoPE, scaling, softmax — all inherited. What it changes is the *cost*: because
nothing older than ``window`` can ever be read again, the cache can drop it. A
full cache grows by one slot per token forever; this one stops growing at
``window``. At an 8,192-token context with a 512 window that is a sixteenth of
the memory, and the saving keeps widening with length.

**What it gives up is reach — though less than it looks.** One layer cannot see
past its window. But layer 2 reads layer 1's outputs, each of which already
summarises a window, so information travels ``window - 1`` positions per layer:
after ``L`` layers a token can be influenced by one ``L × (window - 1)`` behind
it. What is lost is not influence but *retrieval*. Information from far back
arrives only after being blended through every intermediate token, so the model
can know roughly what was said and cannot look up exactly what it was. That is
the failure a few full-attention layers in a hybrid are there to fix.

Reference tier: pure PyTorch, readable, obviously correct. O(n^2) is fine here —
this implementation still builds the whole score matrix and masks most of it
away, so it saves *memory at inference*, not compute at training. Skipping the
masked blocks outright is a kernel's job and belongs in ``mixers/fast/``.

Oracle: full_attention with an explicitly banded mask
Design note: docs/design-notes/sliding-window.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from litterbox.infer.cache import RollingKVCache
from litterbox.model.mixers.base import MixerState
from litterbox.model.mixers.reference.full_attention import FullAttention
from litterbox.model.registry import register_mixer
from litterbox.positional import PositionalEncoding

if TYPE_CHECKING:
    from torch import Tensor


class Windowed:
    """The three things that make an attention mixer a sliding-window one.

    A mixin rather than a subclass of one particular attention class, so the
    reference and fast tiers can both take it: which keys are visible, what the
    cache looks like, and what it costs are the same in either.
    """

    window: int
    _plain_causal = False  # the mask is narrower than "not in the future"

    def _init_window(self, window: int) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.window = window

    def _allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Causal *and* recent: a band ``window`` wide ending on the diagonal."""
        distance = q_pos[:, None] - k_pos[None, :]  # how far behind the query each key sits
        return (distance >= 0) & (distance < self.window)

    def init_state(
        self,
        batch_size: int,
        max_len: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> MixerState:
        """A cache of the last ``window`` tokens. ``max_len`` is ignored — that is the point."""
        cache = RollingKVCache(
            batch_size, self.kv_heads, self.window, self.head_dim, dtype=dtype, device=device
        )
        return MixerState(kv=cache)

    def state_bytes(self, context_len: int) -> float:
        """Grows like full attention up to ``window`` tokens, then not at all."""
        return self.state_bytes_per_token * min(context_len, self.window)


@register_mixer("sliding_window")
class SlidingWindowAttention(Windowed, FullAttention):
    """Local causal attention over the last ``window`` positions.

    Args:
        window: how many tokens a query can see, itself included. ``window=1``
            is a token attending only to itself; ``window >= seq_len`` is full
            attention exactly.

    Everything else is :class:`FullAttention`'s, including the injected
    positional strategy. RoPE suits a window especially well: scores depend on
    distance, and inside a window distances never exceed ``window``, however
    long the document.
    """

    def __init__(
        self,
        d_model: int,
        heads: int,
        kv_heads: int | None = None,
        pos: PositionalEncoding | None = None,
        *,
        window: int,
        bias: bool = False,
    ) -> None:
        super().__init__(d_model, heads, kv_heads, pos, bias=bias)
        self._init_window(window)
