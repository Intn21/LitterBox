"""Rotary position embeddings, with layout as an explicit, named parameter.

Interleaved and non-interleaved layouts rotate different pairs of channels.
Mixing them silently degrades quality instead of failing loudly — the bug
class that bit DeepSeek's DSA indexer. Both layouts are named and tested."""

from __future__ import annotations


def apply_rope(*args, **kwargs):
    """Rotate q/k for the given layout at an absolute position offset."""
    raise NotImplementedError("Milestone 1")
