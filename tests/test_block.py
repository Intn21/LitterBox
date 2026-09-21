"""TransformerBlock: residual wiring, pre-norm, and the promise not to look inside.

The block is nearly all plumbing, so these tests are about the plumbing: what
is added to what, what each half gets to read, and that state and position
reach the mixer untouched.
"""

import math

import pytest
import torch
import torch.nn as nn

from litterbox.model import (
    MixerState,
    RMSNorm,
    SwiGLU,
    TokenMixer,
    TransformerBlock,
    get_mixer,
    scale_residual_projections,
)

D_MODEL, HEADS, SEQ = 32, 4, 12


def real_block():
    return TransformerBlock(
        get_mixer("full_attention")(D_MODEL, HEADS, 2),
        SwiGLU(D_MODEL),
        RMSNorm(D_MODEL),
        RMSNorm(D_MODEL),
    )


class SpyMixer(TokenMixer):
    """A mixer the block has never heard of. Records what it was handed and
    returns a constant, so the tests can see exactly what the block did."""

    def __init__(self, fill=0.0):
        super().__init__()
        self.fill = fill
        self.seen = {}

    def forward(self, x, state=None, pos_offset=0):
        self.seen = {"x": x.detach().clone(), "state": state, "pos_offset": pos_offset}
        return torch.full_like(x, self.fill), state

    @property
    def state_bytes_per_token(self):
        return 0.0


def test_shape_and_state_default():
    out, state = real_block()(torch.randn(2, SEQ, D_MODEL))
    assert out.shape == (2, SEQ, D_MODEL)
    assert state is None


def test_each_half_adds_to_the_stream_rather_than_replacing_it():
    """x -> x + mixer(...) -> that + mlp(...). With constant branch outputs the
    arithmetic is visible: the input survives, plus one contribution each."""
    x = torch.randn(2, SEQ, D_MODEL)
    block = TransformerBlock(SpyMixer(fill=2.0), nn.Identity(), RMSNorm(D_MODEL), nn.Identity())
    # mixer adds 2; then mlp and its norm are identity, so the stream doubles.
    out, _ = block(x)
    assert torch.allclose(out, (x + 2.0) * 2)


def test_silent_branches_make_the_block_an_identity():
    """Zero the last projection of both branches and the block must pass its
    input through exactly — the property that lets a deep stack start out
    behaving like a shallow one."""
    block = real_block()
    with torch.no_grad():
        block.mixer.o_proj.weight.zero_()
        block.mlp.down_proj.weight.zero_()
    x = torch.randn(2, SEQ, D_MODEL)
    assert torch.equal(block(x)[0], x)


def test_pre_norm_the_branch_reads_a_normalised_copy_but_the_stream_is_left_alone():
    spy = SpyMixer(fill=0.0)
    block = TransformerBlock(spy, nn.Identity(), RMSNorm(D_MODEL), nn.Identity())
    x = torch.randn(2, SEQ, D_MODEL) * 100.0  # a loud stream

    out, _ = block(x)

    seen_rms = spy.seen["x"].pow(2).mean(-1).sqrt()
    assert torch.allclose(seen_rms, torch.ones_like(seen_rms), atol=1e-2)  # mixer saw unit scale
    # ...while the stream kept its magnitude: mixer added 0, then identity mlp doubled it.
    assert torch.allclose(out, x * 2)


def test_state_and_pos_offset_reach_the_mixer_untouched():
    spy = SpyMixer()
    block = TransformerBlock(spy, SwiGLU(D_MODEL), RMSNorm(D_MODEL), RMSNorm(D_MODEL))
    token = MixerState()
    _, returned = block(torch.randn(1, 1, D_MODEL), state=token, pos_offset=41)
    assert spy.seen["state"] is token
    assert spy.seen["pos_offset"] == 41
    assert returned is token


def test_rejects_one_norm_used_twice():
    shared = RMSNorm(D_MODEL)
    with pytest.raises(ValueError, match="same module instance"):
        TransformerBlock(SpyMixer(), SwiGLU(D_MODEL), shared, shared)


def test_gradient_reaches_every_parameter():
    block = real_block()
    block(torch.randn(2, SEQ, D_MODEL))[0].pow(2).sum().backward()
    dead = [n for n, p in block.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == []


# ------------------------------------------------ depth-scaled residual init


def test_scale_residual_projections_shrinks_only_the_flagged_projections():
    torch.manual_seed(0)
    n_layers = 8
    blocks = [
        TransformerBlock(
            get_mixer("full_attention")(128, HEADS),
            SwiGLU(128),
            RMSNorm(128),
            RMSNorm(128),
        )
        for _ in range(n_layers)
    ]
    assert scale_residual_projections(blocks) == 2 * n_layers

    expected = 0.02 / math.sqrt(2 * n_layers)
    for block in blocks:
        assert block.mixer.o_proj.weight.std().item() == pytest.approx(expected, rel=0.1)
        assert block.mlp.down_proj.weight.std().item() == pytest.approx(expected, rel=0.1)
        # Everything that is not a branch's final projection keeps the base scale.
        assert block.mixer.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.1)
        assert block.mlp.up_proj.weight.std().item() == pytest.approx(0.02, rel=0.1)
        assert torch.equal(block.mixer_norm.gain.detach(), torch.ones(128))


def test_scale_residual_projections_ignores_branches_that_did_not_opt_in():
    """The helper finds projections by an opt-in flag, never by inspecting
    types. A mixer that sets no flag is simply left alone."""
    blocks = [
        TransformerBlock(SpyMixer(), nn.Linear(D_MODEL, D_MODEL), nn.Identity(), RMSNorm(D_MODEL))
    ]
    before = blocks[0].mlp.weight.detach().clone()
    assert scale_residual_projections(blocks) == 0
    assert torch.equal(blocks[0].mlp.weight, before)
    assert scale_residual_projections([]) == 0
