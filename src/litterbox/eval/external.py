"""Load and evaluate arbitrary HuggingFace models.

The harness must work on models it did not train — that is how the harness
itself gets validated, by reproducing a published RULER/NIAH number for an
existing small long-context model."""

from __future__ import annotations


def load_hf_model(*args, **kwargs):
    """Load an HF causal LM behind the harness's model interface."""
    raise NotImplementedError("Milestone 0")
