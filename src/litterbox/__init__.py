"""LitterBox: a sandbox for long-context token mixers.

Kept deliberately thin — importing the top-level package should not drag in
torch, transformers, or a training loop. Reach for the submodule you need:
``litterbox.model``, ``litterbox.train``, ``litterbox.eval``.
"""

__version__ = "0.0.1"
