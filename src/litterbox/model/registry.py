"""Mixer registry: the lookup that turns a config string into a module.

A ``layer_pattern`` entry names a mixer (``{mixer: gated_deltanet, heads: 12}``)
and this maps that name to a class. Adding a mixer means adding a file and a
decorator — nothing else in the codebase learns about it.

This module is real rather than stubbed, because it is the plumbing the stubs
hang off: a mixer file has to be importable before it can be implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from litterbox.model.mixers.base import TokenMixer

_REGISTRY: dict[str, type] = {}

T = TypeVar("T")


def register_mixer(name: str) -> Callable[[type[T]], type[T]]:
    """Register a :class:`TokenMixer` subclass under ``name``.

    Usage::

        @register_mixer("sliding_window")
        class SlidingWindowAttention(TokenMixer):
            ...

    Raises:
        ValueError: if ``name`` is already taken. Silently shadowing a mixer
            would make a config mean something different depending on import
            order, which is not a bug anyone should have to find twice.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f"mixer {name!r} is already registered to "
                f"{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_mixer(name: str) -> type[TokenMixer]:
    """Look up a registered mixer class.

    Raises:
        KeyError: naming the available mixers. A typo in a config should not
            surface as an opaque lookup failure halfway into building a model.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown mixer {name!r}; available: {', '.join(available_mixers()) or '(none)'}"
        ) from None


def available_mixers() -> list[str]:
    """Names of every registered mixer, sorted."""
    return sorted(_REGISTRY)
