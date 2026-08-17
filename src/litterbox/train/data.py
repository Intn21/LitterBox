"""Streaming tokenized corpus with a sequence-length curriculum.

The curriculum matters for progressive length extension: the same run may
start at 2K and step up to 32K on a schedule."""

from __future__ import annotations


def build_dataloader(*args, **kwargs):
    """Build a streaming loader for a corpus and length schedule."""
    raise NotImplementedError("Milestone 1")
