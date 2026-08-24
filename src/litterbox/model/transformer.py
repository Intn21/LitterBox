"""Backbone: embeddings, the stack of blocks, final norm, and LM head.

The stack is built by tiling ``layer_pattern`` to ``n_layers``, so a 3:1
hybrid and a dense baseline differ only in a config file."""

from __future__ import annotations


def build_model(*args, **kwargs):
    """Instantiate a model from a validated model config."""
    raise NotImplementedError("Milestone 1: backbone lands with the first working mixer.")
