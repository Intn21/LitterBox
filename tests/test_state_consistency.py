"""Parallel/recurrent duality: training-mode and decode-mode must agree.

The full-sequence parallel (or chunked) forward and the step-by-step
recurrence with MixerState must produce the same outputs. This is the single
most bug-prone property of the DeltaNet family and the cheapest test to keep
green from day one."""

import pytest

pytestmark = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)


def test_prefill_equals_sequential_decode():
    """One parallel pass over N tokens == N single-token steps with state."""
    raise NotImplementedError("Milestone 1")


def test_partial_prefill_then_decode():
    """Prefill k tokens, decode the rest; must equal a full parallel pass."""
    raise NotImplementedError("Milestone 1")


def test_chunked_forward_matches_recurrent():
    """Chunked linear-attention forward vs the pure recurrence."""
    raise NotImplementedError("Milestone 3")


def test_state_bytes_per_token_is_honest():
    """Reported bytes/token must match measured state growth over context."""
    raise NotImplementedError("Milestone 1")
