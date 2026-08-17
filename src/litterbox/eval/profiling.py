"""Prefill/decode latency, KV bytes per token, and peak memory.

Reads ``state_bytes_per_token`` off each mixer, so a hybrid's memory profile
falls out of the layer pattern with no per-technique special-casing."""

from __future__ import annotations


def profile_model(*args, **kwargs):
    """Measure latency and memory across a sweep of context lengths."""
    raise NotImplementedError("Milestone 0")
