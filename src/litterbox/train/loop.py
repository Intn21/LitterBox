"""Single-GPU training loop, written by hand rather than delegated.

Deliberately simple: this is the loop whose loss curve gets compared
against nanoGPT on identical data and hyperparameters."""

from __future__ import annotations


def train(*args, **kwargs):
    """Run a training job from a resolved config."""
    raise NotImplementedError("Milestone 1")
