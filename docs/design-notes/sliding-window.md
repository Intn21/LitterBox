# Sliding-window attention

`src/litterbox/model/mixers/reference/sliding_window.py` (and its fused twin in
`fast/`) — causal attention where a token sees itself and the `window - 1` tokens
before it.

Papers: Child et al., *Generating Long Sequences with Sparse Transformers* (2019)
and Beltagy et al., *Longformer* (2020) for local attention; Jiang et al.,
*Mistral 7B* (2023) for the rolling cache; Gemma 2 (2024) for interleaving local
and global layers.

## The idea

Compared with full attention it is one extra condition in the mask:

```
full attention:   key j is visible to the query at p   when  j <= p
sliding window:   ...                                  when  p - window < j <= p
```

That condition is the whole mixer. Projections, heads, grouped queries, RoPE,
scaling and softmax are inherited. To make that literally true, attention's mask
is built from absolute positions in one overridable method, `_allowed(q_pos,
k_pos)`, and the two mixers differ there and nowhere else.

## What it buys: a cache that stops growing

Nothing older than `window` can ever be read again, so it need not be kept. A
full KV cache grows by one slot per token for ever; `RollingKVCache` fills to
`window` and stays there. At 8,192 tokens with a 512 window that is a sixteenth
of the memory, and the gap widens with length. `state_bytes(context_len)` is how
a mixer reports this: unbounded for full attention, capped at `window` tokens
here.

This reference tier saves memory **at inference only**. Training still builds the
full `[s, s]` score matrix and masks most of it away, so its compute and
activation memory are those of full attention. Skipping the masked blocks
outright is a kernel's job.

## What it costs: retrieval, not reach

One layer cannot see past its window. But layer 2 reads layer 1's outputs, each
of which already summarises a window, so influence travels `window - 1` positions
per layer: after `L` layers a token can be affected by one `L × (window - 1)`
behind it. `test_reach_grows_by_window_minus_one_per_layer` measures exactly that.

So a windowed model is not blind to the distant past. What it loses is the
ability to *look something up*: information from far back arrives only after
being blended through every token in between. It can know roughly what was said
and cannot fetch exactly what it was. That is the gap a few full-attention layers
in a hybrid exist to close, and why the ratio is a knob: each full layer restores
exact retrieval and brings back a cache that grows.

## Gotchas

**Off by one, in the direction that is hard to see.** `window` counts the token
itself. `distance < window`, not `<=`: a window of 1 is a token attending only to
itself, and a window equal to the sequence length is full attention exactly. Both
limits are tests, because `<=` still trains and still produces a band — one wider
than the config says.

**After eviction, slot `j` is not position `j`.** A full cache can build its mask
from indices, since slot and position coincide. A rolling cache cannot. It reports
`first_position`, and the mask is built from positions. Assume the keys start at
0 and the first `window` tokens decode correctly, then every later token silently
attends to the wrong band. Prefill-versus-stepwise tests only catch this if the
sequence is longer than the window, so the test window is set well inside it.

**`is_causal` is wrong for a band.** The fused kernel's flag means plain causal.
The fast tier uses it only when the mask really is plain causal
(`_plain_causal`), and passes the explicit band otherwise — which also rules out
the FlashAttention backend for windowed layers, leaving the memory-efficient one.

**`pos_offset` is the number of tokens *seen*, not held.** After 1,000 tokens a
cache with a 64 window holds 64 and the next position is 1,000. The cache tracks
`total` separately from `length` for this, and the mixer checks `pos_offset`
against `total`.

**The cache holds one key more than it needs.** A query reads `window - 1` past
keys plus itself, so keeping `window` past tokens leaves one the mask always
hides. Kept for the simpler statement — "the last `window` tokens" — at the cost
of one slot. Trimming it changed no output in a mutation test, only the byte
accounting.

**RoPE suits a window.** Scores depend on distance, and inside a window distances
never exceed `window` however long the document, so the layer never sees a
rotation it was not trained on. It is part of why windowed layers extrapolate to
longer contexts better than full ones.

## What is tested

| Test | Catches |
|---|---|
| `test_mixers_equivalence.py::test_sliding_window_matches_masked_full_attention` | wrong band — against a mask built pair by pair, sharing no code with the mixer |
| `test_sliding_window.py::test_a_token_is_deaf_to_anything_older_than_its_window` | perturbing token `t` moves exactly positions `t .. t + window - 1` |
| `test_sliding_window.py` limits | `window=1` is self-attention; `window >= seq` is full attention |
| `test_state_consistency.py` | parallel pass equals stepwise decode across the point where the cache starts forgetting; claimed state bytes equal measured |
| `test_config.py` hybrid tests | three rolling caches and one full cache generate what no cache generates; only the full layers keep growing |

Mutation-checked: an off-by-one window, a cache one token short, and a mask that
ignores eviction each fail.
