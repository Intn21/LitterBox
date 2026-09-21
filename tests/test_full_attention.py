"""Full attention: the properties that aren't one of the four test families.

Oracle equivalence lives in test_mixers_equivalence.py and causality in
test_causality.py. These cover what is specific to this mixer: grouped-query
bookkeeping, the positional seam, and the ways construction should refuse.
"""

import pytest
import torch

from litterbox.model import available_mixers, get_mixer
from litterbox.model.mixers.reference.full_attention import FullAttention
from litterbox.positional import Learned, NoPE, RoPE

D_MODEL, HEADS, SEQ = 64, 8, 12
HEAD_DIM = D_MODEL // HEADS


def test_registered_and_shape_preserving():
    assert "full_attention" in available_mixers()
    assert get_mixer("full_attention") is FullAttention
    mixer = FullAttention(D_MODEL, HEADS)
    out, state = mixer(torch.randn(2, SEQ, D_MODEL))
    assert out.shape == (2, SEQ, D_MODEL)
    assert state is None


def test_grouped_query_shrinks_kv_projections_and_cache():
    """GQA's whole point: keys and values are projected, and cached, per KV
    head. Queries are untouched."""
    mha = FullAttention(D_MODEL, HEADS)
    gqa = FullAttention(D_MODEL, HEADS, kv_heads=2)
    mqa = FullAttention(D_MODEL, HEADS, kv_heads=1)

    assert mha.kv_heads == HEADS  # the default is plain multi-head
    assert gqa.q_proj.weight.shape == mha.q_proj.weight.shape
    assert gqa.k_proj.weight.shape == (2 * HEAD_DIM, D_MODEL)
    assert gqa.v_proj.weight.shape == (2 * HEAD_DIM, D_MODEL)

    # fp32: 2 tensors (k and v) x kv_heads x head_dim x 4 bytes.
    assert mha.state_bytes_per_token == 2 * HEADS * HEAD_DIM * 4
    assert gqa.state_bytes_per_token == mha.state_bytes_per_token / 4
    assert mqa.state_bytes_per_token == mha.state_bytes_per_token / 8
    assert gqa.half().state_bytes_per_token == 2 * 2 * HEAD_DIM * 2


def test_query_heads_in_a_group_share_one_kv_head():
    """With kv_heads=2, query heads 0-3 read KV head 0 and 4-7 read KV head 1.
    So perturbing KV head 1's projection must leave the first four heads'
    contribution alone — checked by zeroing the output projection's columns
    for heads 4-7 and confirming the output does not move."""
    torch.manual_seed(0)
    mixer = FullAttention(D_MODEL, HEADS, kv_heads=2)
    with torch.no_grad():
        mixer.o_proj.weight[:, 4 * HEAD_DIM :] = 0  # silence query heads 4-7
    x = torch.randn(1, SEQ, D_MODEL)
    before, _ = mixer(x)
    with torch.no_grad():
        mixer.k_proj.weight[HEAD_DIM:] += 1.0  # disturb KV head 1 only
        mixer.v_proj.weight[HEAD_DIM:] += 1.0
    after, _ = mixer(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_first_token_attends_only_to_itself():
    """Position 0 has exactly one allowed key, so its weight is 1 and its
    output is just its own value pushed through the output projection."""
    torch.manual_seed(0)
    mixer = FullAttention(D_MODEL, HEADS)
    x = torch.randn(2, SEQ, D_MODEL)
    out, _ = mixer(x)
    expected = mixer.o_proj(mixer.v_proj(x[:, 0]))
    assert torch.allclose(out[:, 0], expected, atol=1e-6)


def test_rope_makes_the_layer_shift_invariant():
    """RoPE scores depend only on distance and values carry no position, so
    the whole layer's output is the same wherever the sequence sits."""
    torch.manual_seed(0)
    mixer = FullAttention(D_MODEL, HEADS, 2, pos=RoPE(HEAD_DIM, 128, layout="interleaved"))
    x = torch.randn(2, SEQ, D_MODEL)
    here, _ = mixer(x, pos_offset=0)
    there, _ = mixer(x, pos_offset=100)
    assert torch.allclose(here, there, atol=1e-5)


def test_non_rotary_strategies_pass_straight_through():
    """The mixer calls rotate() unconditionally and never asks what it holds.
    NoPE and an additive strategy both leave q and k alone, so they must give
    the same output as passing no strategy at all."""
    x = torch.randn(2, SEQ, D_MODEL)

    def run(pos):
        torch.manual_seed(1)  # identical projections across variants
        return FullAttention(D_MODEL, HEADS, pos=pos)(x)[0]

    base = run(None)
    assert torch.equal(run(NoPE()), base)
    assert torch.equal(run(Learned(D_MODEL, 64)), base)
    assert not torch.allclose(run(RoPE(HEAD_DIM, 64, layout="half")), base)


def test_gradient_reaches_every_projection():
    mixer = FullAttention(D_MODEL, HEADS, kv_heads=2)
    out, _ = mixer(torch.randn(2, SEQ, D_MODEL))
    out.pow(2).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        grad = getattr(mixer, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_rejects_misuse():
    with pytest.raises(ValueError, match="divisible by heads"):
        FullAttention(D_MODEL, 7)
    with pytest.raises(ValueError, match="divisible by kv_heads"):
        FullAttention(D_MODEL, HEADS, kv_heads=3)
    with pytest.raises(TypeError):
        FullAttention(D_MODEL, HEADS, kv_head=2)  # a typo must not be swallowed
    # A rotary strategy built for the wrong width is caught at the first call.
    wrong = FullAttention(D_MODEL, HEADS, pos=RoPE(HEAD_DIM * 2, 64, layout="half"))
    with pytest.raises(ValueError, match="per head"):
        wrong(torch.randn(1, SEQ, D_MODEL))
    with pytest.raises(NotImplementedError, match="step 3"):
        from litterbox.model import MixerState

        FullAttention(D_MODEL, HEADS)(torch.randn(1, 1, D_MODEL), state=MixerState())
