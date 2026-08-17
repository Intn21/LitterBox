"""Needle-in-a-haystack: single-needle, multi-key, and multi-query variants.

Multi-key is the discriminating one. Pure linear mixers can pass single
needle and still fail multi-key badly, which is the signature the hybrid
ablation is looking for."""

from __future__ import annotations


def run_niah(*args, **kwargs):
    """Build haystacks at each length and score needle retrieval."""
    raise NotImplementedError("Milestone 0")
