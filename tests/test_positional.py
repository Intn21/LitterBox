"""Positional seam: uniform hooks, closed forms, and RoPE's invariants.

The RoPE tests check properties, not reference outputs: norms preserved,
scores depending only on relative position, decode-time offsets matching the
full-sequence computation, and — the DeepSeek bug — mixed layouts breaking
shift invariance where nothing else would catch it.
"""

import math

import pytest
import torch

from litterbox.model.transformer import Transformer
from litterbox.positional import (
    NoPE,
    RoPE,
    Sinusoidal,
    available_positional,
    get_positional,
)

D_MODEL, HEAD_DIM, MAX_LEN = 32, 8, 64


def strategies():
    return [
        Sinusoidal(D_MODEL, MAX_LEN),
        Sinusoidal(D_MODEL, MAX_LEN, learnable=True),
        RoPE(HEAD_DIM, MAX_LEN, layout="interleaved"),
        RoPE(HEAD_DIM, MAX_LEN, layout="half"),
        NoPE(),
    ]


# ------------------------------------------------------------------ the seam


def test_every_strategy_flows_through_both_hooks():
    """Callers invoke embed() and rotate() unconditionally; every strategy
    must accept both, acting through at most one."""
    x = torch.randn(2, 16, D_MODEL)
    q = torch.randn(2, 3, 16, HEAD_DIM)
    k = torch.randn(2, 3, 16, HEAD_DIM)
    for enc in strategies():
        assert enc.embed(x).shape == x.shape
        q2, k2 = enc.rotate(q, k)
        assert q2.shape == q.shape and k2.shape == k.shape


def test_registry_round_trip():
    assert set(available_positional()) >= {"sinusoidal", "rope", "nope"}
    assert get_positional("rope") is RoPE
    with pytest.raises(KeyError, match="available:"):
        get_positional("alibi")


# ---------------------------------------------------------------- sinusoidal


def test_sinusoidal_matches_closed_form():
    enc = Sinusoidal(D_MODEL, MAX_LEN)
    x = torch.zeros(1, MAX_LEN, D_MODEL)
    table = enc.embed(x)[0]
    for p, i in [(0, 0), (7, 0), (13, 4), (63, 15)]:
        angle = p / (10000.0 ** (2 * i / D_MODEL))
        assert table[p, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
        assert table[p, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_sinusoidal_learnable_flag():
    fixed = Sinusoidal(D_MODEL, MAX_LEN)
    assert sum(p.numel() for p in fixed.parameters()) == 0

    learnable = Sinusoidal(D_MODEL, MAX_LEN, learnable=True)
    assert sum(p.numel() for p in learnable.parameters()) == MAX_LEN * D_MODEL
    # Starts at the sinusoid, and gradient reaches the table.
    assert torch.equal(learnable.table.detach(), fixed.table)
    learnable.embed(torch.randn(1, 8, D_MODEL)).sum().backward()
    assert learnable.table.grad is not None
    assert torch.all(learnable.table.grad[8:] == 0)  # rows past seq_len unused


def test_sinusoidal_offset_and_bounds():
    enc = Sinusoidal(D_MODEL, MAX_LEN)
    x = torch.zeros(1, 4, D_MODEL)
    shifted = enc.embed(x, pos_offset=10)
    full = enc.embed(torch.zeros(1, MAX_LEN, D_MODEL))
    assert torch.equal(shifted[0], full[0, 10:14])
    with pytest.raises(ValueError, match="max_seq_len"):
        enc.embed(x, pos_offset=MAX_LEN - 2)


# ---------------------------------------------------------------------- rope


@pytest.mark.parametrize("layout", ["interleaved", "half"])
def test_rope_preserves_norms(layout):
    """Rotations don't change vector length, per position, per head."""
    enc = RoPE(HEAD_DIM, MAX_LEN, layout=layout)
    q = torch.randn(2, 3, 16, HEAD_DIM)
    q2, _ = enc.rotate(q, q.clone())
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


@pytest.mark.parametrize("layout", ["interleaved", "half"])
def test_rope_scores_depend_only_on_relative_position(layout):
    """The defining property: shifting q and k together leaves scores fixed."""
    enc = RoPE(HEAD_DIM, MAX_LEN, layout=layout)
    q = torch.randn(1, 2, 16, HEAD_DIM)
    k = torch.randn(1, 2, 16, HEAD_DIM)
    qa, ka = enc.rotate(q, k, pos_offset=0)
    qb, kb = enc.rotate(q, k, pos_offset=13)
    scores_a = qa @ ka.transpose(-2, -1)
    scores_b = qb @ kb.transpose(-2, -1)
    assert torch.allclose(scores_a, scores_b, atol=1e-4)


def test_rope_layout_mismatch_breaks_shift_invariance():
    """The DeepSeek DSA bug, reproduced on purpose: rotate q in one layout and
    k in the other. Nothing crashes, shapes agree, norms are still preserved —
    but scores now change when both positions shift together, which is exactly
    the silent quality degradation the explicit layout parameter exists to
    prevent."""
    inter = RoPE(HEAD_DIM, MAX_LEN, layout="interleaved")
    half = RoPE(HEAD_DIM, MAX_LEN, layout="half")
    q = torch.randn(1, 2, 16, HEAD_DIM)
    k = torch.randn(1, 2, 16, HEAD_DIM)
    qa, _ = inter.rotate(q, k, pos_offset=0)
    _, ka = half.rotate(q, k, pos_offset=0)
    qb, _ = inter.rotate(q, k, pos_offset=13)
    _, kb = half.rotate(q, k, pos_offset=13)
    scores_a = qa @ ka.transpose(-2, -1)
    scores_b = qb @ kb.transpose(-2, -1)
    assert not torch.allclose(scores_a, scores_b, atol=1e-2)


@pytest.mark.parametrize("layout", ["interleaved", "half"])
def test_rope_decode_offset_matches_full_sequence(layout):
    """Rotating a suffix at pos_offset must equal rotating the full sequence
    and slicing — the property incremental decode relies on."""
    enc = RoPE(HEAD_DIM, MAX_LEN, layout=layout)
    x = torch.randn(1, 2, 16, HEAD_DIM)
    full, _ = enc.rotate(x, x)
    suffix, _ = enc.rotate(x[:, :, 11:], x[:, :, 11:], pos_offset=11)
    assert torch.allclose(suffix, full[:, :, 11:], atol=1e-6)


def test_rope_rejects_misuse():
    with pytest.raises(ValueError, match="even"):
        RoPE(7, MAX_LEN, layout="half")
    with pytest.raises(ValueError, match="layout"):
        RoPE(HEAD_DIM, MAX_LEN, layout="rotate_half")
    enc = RoPE(HEAD_DIM, MAX_LEN, layout="half")
    too_long = torch.randn(1, 1, MAX_LEN + 1, HEAD_DIM)
    with pytest.raises(ValueError, match="yarn"):
        enc.rotate(too_long, too_long)
    with pytest.raises(ValueError, match="per head"):
        enc.rotate(torch.randn(1, 16, D_MODEL), torch.randn(1, 16, D_MODEL))


# ------------------------------------------------------------------ backbone


def test_backbone_takes_any_strategy():
    """The additive hook changes logits; rotary and NoPE leave the blockless
    backbone untouched, because their site of action is inside attention."""
    torch.manual_seed(0)
    ids = torch.randint(0, 128, (2, 16))

    def logits(pos):
        torch.manual_seed(1)  # identical embedding init across variants
        return Transformer(128, D_MODEL, pos=pos)(ids)

    base = logits(None)
    assert not torch.allclose(logits(Sinusoidal(D_MODEL, MAX_LEN)), base)
    assert torch.equal(logits(RoPE(HEAD_DIM, MAX_LEN, layout="half")), base)
    assert torch.equal(logits(NoPE()), base)
