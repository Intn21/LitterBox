"""Adapter over wandb / tensorboard, plus plain JSONL for offline runs."""

from __future__ import annotations


def get_logger(*args, **kwargs):
    """Build the logger named by the config."""
    raise NotImplementedError("Milestone 0")
