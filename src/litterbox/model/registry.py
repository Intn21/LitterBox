"""Registries: the lookup that turns a config string into a class.

There is one registry per swappable compartment. A ``layer_pattern`` entry names
a token mixer (``{mixer: gated_deltanet, heads: 12}``), an ``mlp`` block names a
channel mixer, a ``norm`` block names a norm — and each name is resolved here.

Adding an implementation means adding a file and a decorator. Nothing else in the
codebase learns that it exists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from litterbox.model.mixers.base import TokenMixer

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> class lookup for one compartment."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Register a class under ``name``.

        Raises:
            ValueError: if ``name`` is taken. Silently shadowing an entry would
                make a config mean different things depending on import order,
                which is not a bug anyone should have to find twice.
        """

        def decorator(cls: type[T]) -> type[T]:
            existing = self._entries.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"{self.kind} {name!r} is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Look up a registered class.

        Raises:
            KeyError: naming what *is* available. A typo in a config should not
                surface as an opaque failure halfway through building a model.
        """
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; "
                f"available: {', '.join(self.available()) or '(none)'}"
            ) from None

    def available(self) -> list[str]:
        """Every registered name, sorted."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


MIXERS: Registry = Registry("mixer")
CHANNEL_MIXERS: Registry = Registry("channel mixer")
NORMS: Registry = Registry("norm")


# Convenience aliases. `register_mixer` reads better at a class definition than
# `MIXERS.register`, and the token mixer is the one people add most often.
register_mixer = MIXERS.register


def get_mixer(name: str) -> type[TokenMixer]:
    """Look up a registered token mixer class."""
    return MIXERS.get(name)


def available_mixers() -> list[str]:
    """Names of every registered token mixer, sorted."""
    return MIXERS.available()
