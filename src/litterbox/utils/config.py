"""Config loading and schema validation.

Validation is strict on purpose: an unknown key or a layer_pattern that
does not tile cleanly into n_layers should fail at load time, not surface
as a confusing shape error thirty seconds into a run."""

from __future__ import annotations


def load_config(*args, **kwargs):
    """Load, merge, and validate a config into a typed object."""
    raise NotImplementedError("Milestone 0")
