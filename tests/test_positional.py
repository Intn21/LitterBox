"""Positional seam: uniform hooks, closed forms, learned tables, and RoPE's invariants.

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
    Learned,
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
        Learned(D_MODEL, MAX_LEN),
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
    assert set(available_positional()) >= {"sinusoidal", "learned", "rope", "nope"}
    assert get_positional("rope") is RoPE
    assert get_positional("learned") is Learned
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


# ------------------------------------------------------------------- learned


def test_learned_is_a_random_trainable_table():
    """GPT-2's scheme: every row is a parameter, and nothing about order is
    built in — which is what separates it from Sinusoidal(learnable=True)."""
    torch.manual_seed(0)
    enc = Learned(D_MODEL, MAX_LEN)
    assert sum(p.numel() for p in enc.parameters()) == MAX_LEN * D_MODEL
    assert enc.table.requires_grad
    # Small noise at the token-embedding scale, not a unit-scale sinusoid.
    assert enc.table.std().item() == pytest.approx(0.02, rel=0.15)
    assert not torch.allclose(enc.table.detach(), Sinusoidal(D_MODEL, MAX_LEN).table, atol=0.1)
    loud = Learned(D_MODEL, MAX_LEN, init_std=0.5)
    assert loud.table.std().item() == pytest.approx(0.5, rel=0.15)


def test_learned_gradient_reaches_only_the_slots_used():
    """The lookup is a slice, so its gradient is a scatter: a slot that no
    token occupied gets exactly zero, the same property the token embedding
    has for absent tokens."""
    enc = Learned(D_MODEL, MAX_LEN)
    enc.embed(torch.randn(3, 8, D_MODEL), pos_offset=4).sum().backward()
    grad = enc.table.grad
    assert torch.all(grad[:4] == 0) and torch.all(grad[12:] == 0)
    # Each used row collects one unit of gradient per batch element.
    assert torch.all(grad[4:12] == 3)


def test_learned_offset_and_bounds():
    enc = Learned(D_MODEL, MAX_LEN)
    x = torch.zeros(1, 4, D_MODEL)
    assert torch.equal(enc.embed(x, pos_offset=10)[0], enc.table[10:14])
    with pytest.raises(ValueError, match="never trained"):
        enc.embed(x, pos_offset=MAX_LEN - 2)


def test_learned_needs_no_channel_pairing():
    """Sinusoidal needs an even d_model to pair sin with cos; a plain table
    has no such constraint."""
    enc = Learned(7, MAX_LEN)
    assert enc.embed(torch.zeros(1, 5, 7)).shape == (1, 5, 7)
    with pytest.raises(ValueError, match="even"):
        Sinusoidal(7, MAX_LEN)


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


@pytest.mark.parametrize("cast", ["bfloat16", "half", "to_dtype"])
def test_rope_tables_survive_a_model_wide_downcast(cast):
    """``model.to(torch.bfloat16)`` must not take the angle tables with it.
    bf16 cannot hold the cosine of a large angle, and the damage is silent:
    nothing raises, scores just drift, and more so the longer the context."""
    torch.manual_seed(0)
    enc = RoPE(HEAD_DIM, 2048, layout="half")
    reference = RoPE(HEAD_DIM, 2048, layout="half")
    if cast == "bfloat16":
        enc = enc.bfloat16()
    elif cast == "half":
        enc = enc.half()
    else:
        enc = enc.to(torch.bfloat16)

    assert enc.cos.dtype == torch.float32 and enc.sin.dtype == torch.float32
    assert torch.equal(enc.cos, reference.cos)

    q, k = torch.randn(1, 2, 2048, HEAD_DIM), torch.randn(1, 2, 2048, HEAD_DIM)
    (q1, k1), (q2, k2) = enc.rotate(q, k), reference.rotate(q, k)
    assert torch.equal(q1, q2) and torch.equal(k1, k2)


def test_rope_tables_still_follow_the_module_across_devices():
    """The guard pins precision, not placement."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        pytest.skip("no accelerator on this machine")
    enc = RoPE(HEAD_DIM, MAX_LEN, layout="half").to(device)
    assert enc.cos.device.type == device and enc.cos.dtype == torch.float32
    q = torch.randn(1, 2, 16, HEAD_DIM, device=device)
    rotated, _ = enc.rotate(q, q.clone())
    assert rotated.device.type == device


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
    assert not torch.allclose(logits(Learned(D_MODEL, MAX_LEN)), base)
    assert torch.equal(logits(RoPE(HEAD_DIM, MAX_LEN, layout="half")), base)
    assert torch.equal(logits(NoPE()), base)
