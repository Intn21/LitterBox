"""NoPE: no positional encoding at all — the identity at both hooks.

Not a placeholder. Hybrids deliberately use NoPE on their full-attention
layers (Kimi Linear does this; the linear layers' decay carries position),
which is why ``pos`` is a per-layer config field. It is also the cheapest
proof that the positional seam is real: a strategy that overrides nothing
must flow through the same call sites as one that overrides everything.
"""

from __future__ import annotations

from litterbox.positional.base import PositionalEncoding, register_positional


@register_positional("nope")
class NoPE(PositionalEncoding):
    """Inherits identity ``embed`` and ``rotate`` — that absence is the point."""
