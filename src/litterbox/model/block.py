"""TransformerBlock(mixer, mlp) — the mixer is injected, never constructed here.

Keeping construction outside the block is what allows per-layer mixer and
positional choices without the block knowing which technique it holds."""

from __future__ import annotations


def build_block(*args, **kwargs):
    """Assemble one block from a mixer instance and an MLP config."""
    raise NotImplementedError("Milestone 1")
