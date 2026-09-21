"""Causality: no information flows backwards in time.

Perturbation test rather than mask inspection — change the input at position
t and assert every output at a position < t is bit-identical. This catches
leakage a mask-shape assertion would miss, including off-by-one errors in
chunked kernels and state that is updated before it is read."""

import pytest
import torch

from litterbox.model import get_mixer
from litterbox.positional import RoPE

stub = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)

D_MODEL, HEADS, SEQ = 32, 4, 16


def build(name):
    """One small instance of each mixer that has landed. Add a line per mixer."""
    pos = RoPE(D_MODEL // HEADS, 64, layout="half")
    if name in ("full_attention", "full_attention_fast"):
        return get_mixer(name)(D_MODEL, HEADS, 2, pos=pos)
    if name in ("sliding_window", "sliding_window_fast"):
        return get_mixer(name)(D_MODEL, HEADS, 2, pos=pos, window=5)
    raise KeyError(name)


LANDED = ["full_attention", "full_attention_fast", "sliding_window", "sliding_window_fast"]


@pytest.mark.parametrize("name", LANDED)
@pytest.mark.parametrize("t", [1, 7, SEQ - 1])
def test_no_future_leakage_per_mixer(name, t):
    """Perturb token t; outputs at positions < t must not move at all.

    ``torch.equal``, not ``allclose``: a masked score is -inf, softmax makes
    it exactly 0, and 0 times anything finite adds exactly nothing. Any
    difference at all, however small, is information from the future."""
    torch.manual_seed(0)
    mixer = build(name)
    x = torch.randn(2, SEQ, D_MODEL)
    poked = x.clone()
    poked[:, t] += torch.randn(2, D_MODEL) * 10.0

    before, _ = mixer(x)
    after, _ = mixer(poked)

    assert torch.equal(before[:, :t], after[:, :t])
    # ...and the perturbation must actually reach position t onward, or the
    # test would pass on a mixer that ignores its input.
    assert not torch.allclose(before[:, t:], after[:, t:])


@pytest.mark.parametrize("t", [1, 9, 23])
def test_no_future_leakage_through_a_full_model(assemble, t):
    """The same perturbation test, from token ids to logits, through
    embeddings, two full blocks, the final norm, and the LM head. Only
    attention is allowed to move information between positions; this proves
    nothing else in the stack does it by accident."""
    torch.manual_seed(0)
    model = assemble(vocab=64)
    ids = torch.randint(0, 64, (2, 24))
    poked = ids.clone()
    poked[:, t] = (poked[:, t] + 1) % 64

    before, after = model(ids), model(poked)

    assert torch.equal(before[:, :t], after[:, :t])
    assert not torch.allclose(before[:, t:], after[:, t:])


@stub
def test_no_leakage_across_chunk_boundaries():
    """Chunked implementations must not leak across the chunk seam."""
    raise NotImplementedError("Milestone 3")
