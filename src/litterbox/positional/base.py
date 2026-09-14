"""The positional-encoding seam: one interface, two hooks, a name registry.

Positional strategies disagree about *where* they act, not just how. Additive
schemes (sinusoidal, learned) modify the residual stream once, at the
embedding. Rotary schemes modify queries and keys inside every attention
layer. NoPE modifies nothing. One interface can hold all three only if it
exposes both sites:

- :meth:`PositionalEncoding.embed` — called by the backbone on ``[B, S, D]``
  activations right after token embedding.
- :meth:`PositionalEncoding.rotate` — called by an attention mixer on
  ``[..., S, head_dim]`` queries and keys.

Both default to identity, so a strategy implements only the hook it uses and
callers invoke both unconditionally — no ``isinstance`` checks, which is the
same rule the block obeys for mixers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Base strategy: identity at both hooks.

    ``pos_offset`` is the absolute position of index 0 of the tensor being
    processed — 0 during training on full sequences, and the current length
    during incremental decode. Every hook takes it; strategies that ignore it
    (learned tables would not, RoPE must not) simply don't read it.
    """

    def embed(self, x: Tensor, pos_offset: int = 0) -> Tensor:
        """Additive hook, applied once to ``[batch, seq, d_model]``."""
        return x

    def rotate(self, q: Tensor, k: Tensor, pos_offset: int = 0) -> tuple[Tensor, Tensor]:
        """Q/K hook, applied per attention layer to ``[..., seq, head_dim]``."""
        return q, k


_REGISTRY: dict[str, type] = {}

T = TypeVar("T")


def register_positional(name: str) -> Callable[[type[T]], type[T]]:
    """Register a :class:`PositionalEncoding` subclass under ``name``.

    Raises:
        ValueError: if ``name`` is already taken, for the same reason the mixer
            registry does — a config must not change meaning with import order.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"positional encoding {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_positional(name: str) -> type[PositionalEncoding]:
    """Look up a registered positional encoding class by config name.

    Raises:
        KeyError: naming the available strategies.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown positional encoding {name!r}; "
            f"available: {', '.join(available_positional()) or '(none)'}"
        ) from None


def available_positional() -> list[str]:
    """Names of every registered positional encoding, sorted."""
    return sorted(_REGISTRY)
