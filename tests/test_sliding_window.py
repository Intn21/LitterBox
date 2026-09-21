"""Sliding-window attention: what is specific to a window.

Oracle equivalence, causality and parallel-versus-stepwise live in the shared
test families. These cover the window itself: locality, the limit where it
becomes full attention, how reach grows with depth, and the cache that forgets.
"""

import pytest
import torch

from litterbox.infer import RollingKVCache
from litterbox.model import get_mixer
from litterbox.model.mixers.reference.full_attention import FullAttention
from litterbox.model.mixers.reference.sliding_window import SlidingWindowAttention
from litterbox.positional import RoPE

D_MODEL, HEADS, SEQ = 32, 4, 24
HEAD_DIM = D_MODEL // HEADS
TIERS = ["sliding_window", "sliding_window_fast"]


def make(name, window, kv_heads=2):
    torch.manual_seed(0)
    return get_mixer(name)(
        D_MODEL, HEADS, kv_heads, pos=RoPE(HEAD_DIM, 128, layout="half"), window=window
    )


@pytest.mark.parametrize("name", TIERS)
def test_a_token_is_deaf_to_anything_older_than_its_window(name):
    """The defining property, by perturbation: poke token t and only positions
    t .. t + window - 1 may change. Earlier is causality; later is locality."""
    window, t = 5, 8
    mixer = make(name, window)
    x = torch.randn(2, SEQ, D_MODEL)
    poked = x.clone()
    poked[:, t] += 10.0
    before, after = mixer(x)[0], mixer(poked)[0]

    moved = ~torch.isclose(before, after, atol=1e-6).all(dim=-1).all(dim=0)  # per position
    assert moved.nonzero().flatten().tolist() == list(range(t, t + window))
    assert torch.equal(before[:, :t], after[:, :t])  # the past: bit-identical


@pytest.mark.parametrize("name", TIERS)
def test_a_window_as_long_as_the_sequence_is_full_attention(name):
    windowed = make(name, SEQ)
    torch.manual_seed(0)
    full = FullAttention(D_MODEL, HEADS, 2, pos=RoPE(HEAD_DIM, 128, layout="half"))
    full.load_state_dict(windowed.state_dict())  # same parameters, same keys
    x = torch.randn(2, SEQ, D_MODEL)
    assert torch.allclose(windowed(x)[0], full(x)[0], atol=1e-5)
    assert not torch.allclose(make(name, 3)(x)[0], full(x)[0], atol=1e-3)


def test_window_of_one_attends_only_to_itself():
    mixer = make("sliding_window", 1, kv_heads=HEADS)  # multi-head, so values need no repeating
    x = torch.randn(2, SEQ, D_MODEL)
    assert torch.allclose(mixer(x)[0], mixer.o_proj(mixer.v_proj(x)), atol=1e-6)


def test_reach_grows_by_window_minus_one_per_layer():
    """One layer cannot see past its window, but layer 2 reads layer 1's outputs,
    each of which already summarises a window. Influence travels window - 1
    positions per layer — which is why a windowed model is not blind to the
    distant past, only unable to look it up."""
    window = 4
    layers = [make("sliding_window", window) for _ in range(3)]
    with torch.no_grad():
        # Init-scale weights (0.02) attenuate a signal ~1000x per layer, which
        # after three layers is below fp32 resolution: the influence is there
        # and cannot be measured. Trained-scale weights make it visible.
        for layer in layers:
            for p in layer.parameters():
                p.mul_(25.0)
    x = torch.randn(1, SEQ, D_MODEL)
    poked = x.clone()
    poked[:, 0] += 10.0

    a, b = x, poked
    for depth, layer in enumerate(layers, start=1):
        a, b = a + layer(a)[0], b + layer(b)[0]  # residual, as in a block
        changed = ~torch.isclose(a, b, atol=1e-6).all(dim=-1)[0]
        assert int(changed.nonzero().max()) == depth * (window - 1)


def test_state_is_bounded_and_says_so():
    mixer = make("sliding_window", 6)
    per_token = mixer.state_bytes_per_token
    assert [mixer.state_bytes(n) for n in (1, 6, 7, 10_000)] == [
        per_token,
        6 * per_token,
        6 * per_token,
        6 * per_token,
    ]
    full = FullAttention(D_MODEL, HEADS, 2)
    assert full.state_bytes(10_000) == 10_000 * per_token  # the curve it is an answer to

    state = mixer.init_state(1, max_len=10_000)  # max_len is ignored: the state is bounded
    assert isinstance(state.kv, RollingKVCache) and state.kv.window == 6


def test_window_is_required_and_validated():
    with pytest.raises(TypeError, match="window"):
        SlidingWindowAttention(D_MODEL, HEADS)
    with pytest.raises(ValueError, match="at least 1"):
        SlidingWindowAttention(D_MODEL, HEADS, window=0)


# --------------------------------------------------------- the rolling cache


def kv(n, start=0):
    """Keys whose values are their own absolute position, so eviction is visible."""
    k = torch.arange(start, start + n, dtype=torch.float32).view(1, 1, n, 1).expand(2, 3, n, 4)
    return k.clone(), k.clone()


def test_rolling_cache_keeps_the_last_window_tokens_and_reports_where_they_start():
    cache = RollingKVCache(2, 3, window=4, head_dim=4)

    k_all, _ = cache.append(*kv(3))
    assert k_all[0, 0, :, 0].tolist() == [0, 1, 2] and cache.first_position == 0
    assert (cache.length, cache.total) == (3, 3)

    k_all, _ = cache.append(*kv(3, start=3))  # the kept past, then the new tokens
    assert k_all[0, 0, :, 0].tolist() == [0, 1, 2, 3, 4, 5] and cache.first_position == 0
    assert (cache.length, cache.total) == (4, 6)  # 0 and 1 have been dropped

    k_all, _ = cache.append(*kv(1, start=6))
    assert k_all[0, 0, :, 0].tolist() == [2, 3, 4, 5, 6]
    assert cache.first_position == 2  # slot 0 no longer holds position 0
    assert (cache.length, cache.total) == (4, 7)


def test_rolling_cache_memory_stops_growing():
    cache = RollingKVCache(1, 3, window=8, head_dim=4)
    used = []
    for t in range(40):
        cache.append(*[x[:1] for x in kv(1, start=t)])
        used.append(cache.nbytes_used)
    assert used[7] == 8 * cache.bytes_per_token
    assert set(used[7:]) == {used[7]}  # flat from there on

    cache.reset()
    assert (cache.length, cache.total, cache.first_position) == (0, 0, 0)
    with pytest.raises(ValueError, match="before they are repeated"):
        cache.append(torch.randn(1, 6, 1, 4), torch.randn(1, 6, 1, 4))
    with pytest.raises(ValueError, match="at least 1"):
        RollingKVCache(1, 3, window=0, head_dim=4)
