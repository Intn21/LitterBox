"""Oracle equivalence: every mixer matches a trusted reference numerically.

Full attention is checked against PyTorch SDPA, the DeltaNet family against
flash-linear-attention, and each fast/ implementation against its reference
twin. Discrepancies here are where the learning is, so tolerances should be
tight enough to be informative rather than merely green."""

import pytest

pytestmark = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)


def test_full_attention_matches_sdpa():
    """Reference MHA/GQA vs torch.nn.functional.scaled_dot_product_attention."""
    raise NotImplementedError("Milestone 1")


def test_sliding_window_matches_masked_full_attention():
    """SWA vs full attention under an explicitly banded mask."""
    raise NotImplementedError("Milestone 2")


@pytest.mark.oracle
def test_deltanet_matches_fla():
    """DeltaNet vs flash-linear-attention. fla is a test dependency only."""
    raise NotImplementedError("Milestone 3")


@pytest.mark.oracle
def test_gated_deltanet_matches_fla():
    """Gated DeltaNet vs flash-linear-attention."""
    raise NotImplementedError("Milestone 3")


def test_dsa_with_full_topk_matches_dense():
    """With k = full context, selection is a no-op and DSA must equal dense."""
    raise NotImplementedError("Milestone 4")
