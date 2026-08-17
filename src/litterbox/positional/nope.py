"""NoPE: no positional encoding at all — the identity.

Not a placeholder. Hybrids deliberately use NoPE on their full-attention
layers (Kimi Linear does this; the linear layers' decay carries position),
which is why ``pos`` is a per-layer config field."""

from __future__ import annotations


def apply_nope(*args, **kwargs):
    """Return q/k unchanged. Exists so per-layer positional config is total."""
    raise NotImplementedError("Milestone 2")
