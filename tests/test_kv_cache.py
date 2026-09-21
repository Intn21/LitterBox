"""KVCache: what it stores, what it costs, and what it refuses."""

import pytest
import torch

from litterbox.infer import KVCache

B, KV, MAX, HD = 2, 3, 16, 8


def kv(n, fill=None):
    k, v = torch.randn(B, KV, n, HD), torch.randn(B, KV, n, HD)
    return (k.fill_(fill), v.fill_(fill)) if fill is not None else (k, v)


def test_append_returns_everything_stored_so_far_in_order():
    cache = KVCache(B, KV, MAX, HD)
    k1, v1 = kv(5)
    k2, v2 = kv(1)
    cache.append(k1, v1)
    k_all, v_all = cache.append(k2, v2)
    assert cache.length == 6 and k_all.shape == (B, KV, 6, HD)
    assert torch.equal(k_all, torch.cat((k1, k2), dim=2))
    assert torch.equal(v_all, torch.cat((v1, v2), dim=2))


def test_append_hands_back_a_view_not_a_copy():
    """The whole point of preallocating: no step copies what is already stored."""
    cache = KVCache(B, KV, MAX, HD)
    k_all, _ = cache.append(*kv(4))
    assert k_all.data_ptr() == cache.k.data_ptr()
    assert not k_all.is_contiguous() or k_all.shape[2] == MAX  # a slice of the big buffer


def test_cost_is_per_kv_head_and_grows_only_with_context():
    cache = KVCache(B, KV, MAX, HD)
    assert cache.bytes_per_token == 2 * KV * HD * 4
    assert cache.nbytes_used == 0
    cache.append(*kv(7))
    assert cache.nbytes_used == B * 7 * cache.bytes_per_token
    assert cache.nbytes_allocated == 2 * B * KV * MAX * HD * 4  # fixed, whatever is stored
    assert (
        KVCache(B, KV, MAX, HD, dtype=torch.float16).bytes_per_token == cache.bytes_per_token // 2
    )
    assert "7/16 tokens" in repr(cache)


def test_a_full_cache_raises_rather_than_dropping_the_oldest_token():
    cache = KVCache(B, KV, 4, HD)
    cache.append(*kv(3))
    with pytest.raises(ValueError, match="cache is full: 3 stored \\+ 2 new > max_len=4"):
        cache.append(*kv(2))
    assert cache.length == 3  # the failed write changed nothing
    cache.append(*kv(1))


def test_shape_mismatches_are_refused():
    cache = KVCache(B, KV, MAX, HD)
    with pytest.raises(ValueError, match="before they are repeated"):
        cache.append(torch.randn(B, KV * 2, 1, HD), torch.randn(B, KV * 2, 1, HD))
    with pytest.raises(ValueError, match="shapes differ"):
        cache.append(torch.randn(B, KV, 1, HD), torch.randn(B, KV, 2, HD))


def test_reset_forgets_the_tokens_and_keeps_the_allocation():
    cache = KVCache(B, KV, MAX, HD)
    cache.append(*kv(9))
    ptr = cache.k.data_ptr()
    cache.reset()
    assert cache.length == 0 and cache.nbytes_used == 0 and cache.k.data_ptr() == ptr
    k_all, _ = cache.append(*kv(2, fill=1.0))
    assert k_all.shape[2] == 2 and bool((k_all == 1.0).all())
