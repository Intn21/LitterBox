"""RMSNorm and SwiGLU: oracles, defining properties, and the per-token guarantee.

Both layers are position-wise — they act on each token's vector alone. That is
what lets a block stay causal: only the token mixer may move information
between positions, so everything else must provably move none.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from litterbox.model import RMSNorm, SwiGLU
from litterbox.model.mlp import swiglu_hidden_dim

D_MODEL, SEQ = 48, 10


# ------------------------------------------------------------------- RMSNorm


def test_rmsnorm_matches_torch():
    torch.manual_seed(0)
    ours, ref = RMSNorm(D_MODEL, eps=1e-5), nn.RMSNorm(D_MODEL, eps=1e-5)
    with torch.no_grad():
        gain = torch.randn(D_MODEL)
        ours.gain.copy_(gain)
        ref.weight.copy_(gain)
    x = torch.randn(3, SEQ, D_MODEL) * 7.0
    assert torch.allclose(ours(x), ref(x), atol=1e-6)


def test_rmsnorm_output_has_unit_rms_whatever_the_input_scale():
    """The whole job: the next layer sees the same scale no matter what."""
    norm = RMSNorm(D_MODEL)
    for scale in (0.1, 1.0, 1e3):
        y = norm(torch.randn(4, SEQ, D_MODEL) * scale)
        rms = y.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_rmsnorm_eps_sets_a_floor_on_what_gets_normalized():
    """eps sits under the square root next to mean(x^2), so once activations
    shrink to around sqrt(eps) it dominates and the output is no longer unit
    scale. With eps=1e-5 that floor is ~3e-3. Not a bug — it is what keeps a
    zero vector finite — but it means eps is a statement about the smallest
    activations you expect, not just a guard against dividing by zero."""
    tiny = torch.randn(4, SEQ, D_MODEL) * 1e-3  # mean square ~1e-6, below eps

    def out_rms(eps):
        return RMSNorm(D_MODEL, eps=eps)(tiny).pow(2).mean(-1).sqrt().mean().item()

    assert out_rms(1e-5) < 0.5  # eps dominates: visibly under-normalized
    assert out_rms(1e-12) == pytest.approx(1.0, abs=1e-2)


def test_rmsnorm_ignores_scale_but_not_shift():
    """Multiplying the input by a constant changes nothing. Adding a constant
    does — RMSNorm has no mean subtraction, which is exactly how it differs
    from LayerNorm."""
    torch.manual_seed(0)
    x = torch.randn(2, SEQ, D_MODEL)
    rms, layer = RMSNorm(D_MODEL, eps=1e-8), nn.LayerNorm(D_MODEL, eps=1e-8)

    assert torch.allclose(rms(x * 50.0), rms(x), atol=1e-5)
    assert not torch.allclose(rms(x + 3.0), rms(x), atol=1e-2)
    assert torch.allclose(layer(x + 3.0), layer(x), atol=1e-4)  # LayerNorm re-centers


def test_rmsnorm_gain_starts_at_one_and_learns():
    norm = RMSNorm(D_MODEL)
    assert torch.equal(norm.gain.detach(), torch.ones(D_MODEL))
    assert [n for n, _ in norm.named_parameters()] == ["gain"]  # no bias
    norm(torch.randn(2, SEQ, D_MODEL)).pow(2).sum().backward()
    assert norm.gain.grad is not None and norm.gain.grad.abs().sum() > 0


def test_rmsnorm_zero_vector_stays_finite():
    y = RMSNorm(D_MODEL)(torch.zeros(1, 2, D_MODEL))
    assert torch.equal(y, torch.zeros_like(y))


def test_rmsnorm_keeps_dtype_but_reduces_in_fp32():
    """Squaring 300 in fp16 overflows to inf. The reduction runs in fp32, so
    the half-precision output is still finite and correct."""
    norm = RMSNorm(D_MODEL).half()
    x = torch.full((1, 1, D_MODEL), 300.0, dtype=torch.float16)
    assert x.pow(2).isinf().all()  # the naive computation would already be lost
    y = norm(x)
    assert y.dtype == torch.float16
    assert torch.allclose(y.float(), torch.ones(1, 1, D_MODEL), atol=1e-3)


# -------------------------------------------------------------------- SwiGLU


def test_swiglu_matches_the_silu_oracle():
    torch.manual_seed(0)
    mlp = SwiGLU(D_MODEL)
    x = torch.randn(3, SEQ, D_MODEL)
    ref = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
    assert torch.allclose(mlp(x), ref, atol=1e-6)
    assert mlp(x).shape == x.shape


def test_swiglu_hidden_width_matches_a_plain_mlp_budget():
    """Three matrices at 2/3 the width cost the same as two at full width."""
    assert swiglu_hidden_dim(768, 4) == 2048  # exactly 8/3 * 768, already a multiple of 64
    assert swiglu_hidden_dim(100, 4) == 320  # 266.67 rounded up to a multiple of 64
    assert swiglu_hidden_dim(100, 4, multiple_of=1) == 266

    mlp = SwiGLU(768, hidden_mult=4)
    plain_mlp_params = 2 * 768 * (4 * 768)
    assert sum(p.numel() for p in mlp.parameters()) == plain_mlp_params
    assert SwiGLU(768, hidden_dim=1000).hidden_dim == 1000


def test_swiglu_has_three_matrices_and_no_biases():
    names = sorted(n for n, _ in SwiGLU(D_MODEL).named_parameters())
    assert names == ["down_proj.weight", "gate_proj.weight", "up_proj.weight"]
    with_bias = sorted(n for n, _ in SwiGLU(D_MODEL, bias=True).named_parameters())
    assert len(with_bias) == 6


def test_swiglu_gate_can_close_a_channel():
    """Drive every gate pre-activation far negative: silu goes to ~0 and the
    content is blocked, whatever it holds."""
    mlp = SwiGLU(D_MODEL, bias=True)
    x = torch.randn(2, SEQ, D_MODEL)
    open_out = mlp(x)
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()
        mlp.gate_proj.bias.fill_(-30.0)
    assert open_out.abs().max() > 1e-4
    assert mlp(x).abs().max() < 1e-8


def test_swiglu_gradient_reaches_all_three_matrices():
    mlp = SwiGLU(D_MODEL)
    mlp(torch.randn(2, SEQ, D_MODEL)).pow(2).sum().backward()
    for name in ("gate_proj", "up_proj", "down_proj"):
        grad = getattr(mlp, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, name


# ------------------------------------------------- both: strictly per token


@pytest.mark.parametrize("make", [lambda: RMSNorm(D_MODEL), lambda: SwiGLU(D_MODEL)])
def test_no_information_moves_between_positions(make):
    """Perturb one token; every other token's output must be bit-identical,
    in both directions. If this ever fails, a 'position-wise' layer is leaking
    across positions and the block's causality guarantee is gone."""
    torch.manual_seed(0)
    layer = make()
    x = torch.randn(2, SEQ, D_MODEL)
    poked = x.clone()
    poked[:, 4] += 10.0
    before, after = layer(x), layer(poked)
    assert torch.equal(before[:, :4], after[:, :4])
    assert torch.equal(before[:, 5:], after[:, 5:])
    assert not torch.allclose(before[:, 4], after[:, 4])
