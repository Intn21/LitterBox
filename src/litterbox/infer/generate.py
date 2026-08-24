"""Greedy and sampled generation, driving every mixer through MixerState.

Generation code never branches on mixer type: whether a layer carries a KV
cache or a recurrent matrix is the mixer's business."""

from __future__ import annotations


def generate(*args, **kwargs):
    """Generate tokens from a prompt, threading state through the stack."""
    raise NotImplementedError("Milestone 1")
