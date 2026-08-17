"""Causality: no information flows backwards in time.

Perturbation test rather than mask inspection — change the input at position
t and assert every output at a position < t is bit-identical. This catches
leakage a mask-shape assertion would miss, including off-by-one errors in
chunked kernels and state that is updated before it is read."""

import pytest

pytestmark = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)


def test_no_future_leakage_per_mixer():
    """Perturb token t; outputs at positions < t must not move at all."""
    raise NotImplementedError("Milestone 1")


def test_no_leakage_across_chunk_boundaries():
    """Chunked implementations must not leak across the chunk seam."""
    raise NotImplementedError("Milestone 3")
