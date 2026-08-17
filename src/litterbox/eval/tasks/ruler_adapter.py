"""Adapter around the official RULER suite.

Wrapped, never reimplemented: a reimplementation would silently drift from
the published numbers this harness is validated against."""

from __future__ import annotations


def run_ruler(*args, **kwargs):
    """Run RULER tasks at the configured lengths and normalize the output."""
    raise NotImplementedError("Milestone 0")
