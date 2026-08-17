"""Thin adapter to an external distributed backend (torchtitan / accelerate).

Intentionally empty until single-GPU work has outgrown itself — everything
through Milestone 4 fits on one 24-80GB card, and building this early is a
documented way for the framework to become the project."""

from __future__ import annotations


def wrap_for_distributed(*args, **kwargs):
    """Wrap a model and loop for multi-device execution."""
    raise NotImplementedError("Milestone 5: do not build this earlier.")
