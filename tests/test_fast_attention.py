"""The fast tier is tested against its reference twin, never against the paper.

Same weights in, same numbers out — forward and backward — and the same
state-dict keys, so a checkpoint moves freely between tiers.
"""

import pytest
import torch

from litterbox.model import dense_transformer, get_mixer
from litterbox.positional import NoPE, RoPE

D_MODEL, HEADS, SEQ = 64, 8, 24
HEAD_DIM = D_MODEL // HEADS


def twins(kv_heads, pos):
    def strategy():
        return NoPE() if pos == "nope" else RoPE(HEAD_DIM, 128, layout="half")

    torch.manual_seed(0)
    reference = get_mixer("full_attention")(D_MODEL, HEADS, kv_heads, pos=strategy())
    fast = get_mixer("full_attention_fast")(D_MODEL, HEADS, kv_heads, pos=strategy())
    fast.load_state_dict(reference.state_dict())  # strict: the keys must already agree
    return reference, fast


@pytest.mark.parametrize("kv_heads", [8, 2, 1], ids=["mha", "gqa", "mqa"])
@pytest.mark.parametrize("pos", ["nope", "rope"])
def test_fast_matches_reference_forward_and_backward(kv_heads, pos):
    reference, fast = twins(kv_heads, pos)
    x = torch.randn(3, SEQ, D_MODEL)

    out_ref, _ = reference(x)
    out_fast, state = fast(x)
    assert state is None
    assert torch.allclose(out_fast, out_ref, atol=1e-5)

    out_ref.pow(2).sum().backward()
    out_fast.pow(2).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        g_ref, g_fast = getattr(reference, name).weight.grad, getattr(fast, name).weight.grad
        assert torch.allclose(g_fast, g_ref, atol=1e-4), name


def test_tiers_share_parameters_and_cache_accounting():
    reference, fast = twins(2, "rope")
    assert list(reference.state_dict()) == list(fast.state_dict())
    assert reference.state_bytes_per_token == fast.state_bytes_per_token
    assert fast.o_proj.residual_out is True  # still opts in to depth-scaled init


def test_fast_honours_pos_offset_like_the_reference():
    reference, fast = twins(2, "rope")
    x = torch.randn(2, SEQ, D_MODEL)
    assert torch.allclose(fast(x, pos_offset=40)[0], reference(x, pos_offset=40)[0], atol=1e-5)


def test_a_whole_model_is_the_same_model_in_either_tier():
    """Swapping the mixer name changes speed and memory, never the numbers —
    and a checkpoint from one tier loads into the other."""
    torch.manual_seed(0)
    slow = dense_transformer(128, D_MODEL, 2, HEADS, 2, max_seq_len=64)
    quick = dense_transformer(
        128, D_MODEL, 2, HEADS, 2, max_seq_len=64, mixer="full_attention_fast"
    )
    quick.load_state_dict(slow.state_dict())
    ids = torch.randint(0, 128, (2, 32))
    assert torch.allclose(quick(ids), slow(ids), atol=1e-4)
