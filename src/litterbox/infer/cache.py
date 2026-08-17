"""Inference state containers: KVCache, RecurrentState, SparseIndexCache.

Each reports its own byte cost so profiling can put a growing KV cache and
a constant-size recurrent state on the same axis."""

from __future__ import annotations


def make_state(*args, **kwargs):
    """Allocate the right state container for a given mixer."""
    raise NotImplementedError("Milestone 1")
