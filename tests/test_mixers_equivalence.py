"""Oracle equivalence: every mixer matches a trusted reference numerically.

Full attention is checked against PyTorch SDPA, the DeltaNet family against
flash-linear-attention, and each fast/ implementation against its reference
twin. Discrepancies here are where the learning is, so tolerances should be
tight enough to be informative rather than merely green."""

import pytest
import torch
import torch.nn.functional as F

from litterbox.model import get_mixer
from litterbox.positional import NoPE, RoPE

# Per-test rather than module-wide, so a mixer that has landed counts as a real
# pass. Remove a marker when its mixer lands.
stub = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)

D_MODEL, HEADS, SEQ = 64, 8, 24


@pytest.mark.parametrize("kv_heads", [8, 2, 1], ids=["mha", "gqa", "mqa"])
@pytest.mark.parametrize("pos", ["nope", "rope"])
def test_full_attention_matches_sdpa(kv_heads, pos):
    """Reference MHA/GQA vs torch.nn.functional.scaled_dot_product_attention.

    The oracle reuses the mixer's projections and positional strategy, then
    hands scoring, masking, softmax, and the KV-head grouping to SDPA. So what
    is under test is the hand-written attention math and the grouping
    convention, which SDPA's ``enable_gqa`` checks independently rather than
    mirroring."""
    torch.manual_seed(0)
    head_dim = D_MODEL // HEADS
    strategy = NoPE() if pos == "nope" else RoPE(head_dim, 128, layout="half")
    mixer = get_mixer("full_attention")(D_MODEL, HEADS, kv_heads, pos=strategy)
    x = torch.randn(3, SEQ, D_MODEL)

    ours, state = mixer(x)
    assert state is None

    q = mixer.q_proj(x).view(3, SEQ, HEADS, head_dim).transpose(1, 2)
    k = mixer.k_proj(x).view(3, SEQ, kv_heads, head_dim).transpose(1, 2)
    v = mixer.v_proj(x).view(3, SEQ, kv_heads, head_dim).transpose(1, 2)
    q, k = strategy.rotate(q, k)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    ref = mixer.o_proj(ref.transpose(1, 2).reshape(3, SEQ, D_MODEL))

    assert torch.allclose(ours, ref, atol=1e-6)


@pytest.mark.parametrize("name", ["sliding_window", "sliding_window_fast"])
@pytest.mark.parametrize("window", [1, 5, SEQ, SEQ * 4])
def test_sliding_window_matches_masked_full_attention(name, window):
    """SWA vs full attention under an explicitly banded mask.

    The band is built here by brute force, one (query, key) pair at a time, so
    it shares no code with the mixer's own mask."""
    torch.manual_seed(0)
    head_dim, kv_heads = D_MODEL // HEADS, 2
    mixer = get_mixer(name)(
        D_MODEL, HEADS, kv_heads, pos=RoPE(head_dim, 128, layout="half"), window=window
    )
    x = torch.randn(2, SEQ, D_MODEL)
    ours, _ = mixer(x)

    band = torch.tensor([[0 <= i - j < window for j in range(SEQ)] for i in range(SEQ)])
    q = mixer.q_proj(x).view(2, SEQ, HEADS, head_dim).transpose(1, 2)
    k = mixer.k_proj(x).view(2, SEQ, kv_heads, head_dim).transpose(1, 2)
    v = mixer.v_proj(x).view(2, SEQ, kv_heads, head_dim).transpose(1, 2)
    q, k = mixer.pos.rotate(q, k)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=band, enable_gqa=True)
    ref = mixer.o_proj(ref.transpose(1, 2).reshape(2, SEQ, D_MODEL))

    assert torch.allclose(ours, ref, atol=1e-5)


@stub
@pytest.mark.oracle
def test_deltanet_matches_fla():
    """DeltaNet vs flash-linear-attention. fla is a test dependency only."""
    raise NotImplementedError("Milestone 3")


@stub
@pytest.mark.oracle
def test_gated_deltanet_matches_fla():
    """Gated DeltaNet vs flash-linear-attention."""
    raise NotImplementedError("Milestone 3")


@stub
def test_dsa_with_full_topk_matches_dense():
    """With k = full context, selection is a no-op and DSA must equal dense."""
    raise NotImplementedError("Milestone 4")
