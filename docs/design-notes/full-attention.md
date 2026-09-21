# Full attention

`src/litterbox/model/mixers/reference/full_attention.py` — multi-head and
grouped-query causal attention with an injected positional strategy.

Papers: Vaswani et al., *Attention Is All You Need* (2017) for the mechanism;
Shazeer, *Fast Transformer Decoding* (2019) for multi-query; Ainslie et al.,
*GQA* (2023) for grouped-query.

## The idea

Each token asks a question and every earlier token offers a label and a payload.

- **query** — what this token is looking for
- **key** — what each token advertises about itself
- **value** — what each token hands over if chosen

A token scores its query against every key, softmaxes the scores into weights
that sum to one, and takes the weighted average of the values. That average is
the token's output. Everything else is bookkeeping around those three lines:

```
scores  = q @ kᵀ / sqrt(head_dim)      how well each key answers each query
weights = softmax(mask(scores))         a distribution over the past
out     = weights @ v                   the past, averaged by relevance
```

**Causal** means position `i` may only use positions `j <= i`. **Multi-head**
means the model width is cut into `heads` narrower slices that each run the
three lines independently, so different heads can look for different things.

## Grouped-query attention

Plain multi-head gives every query head its own key and value head. At decode
time the keys and values of every past token are cached, so the cache costs
`2 × heads × head_dim` numbers per token per layer, and that cache — not the
weights — is what runs out of memory at long context.

GQA keeps all the query heads but shares each key/value head among a group of
them. With `heads=8, kv_heads=2`, query heads 0–3 read KV head 0 and heads 4–7
read KV head 1. The cache shrinks by `heads / kv_heads` (4× here); the number
of distinct questions asked does not shrink at all. `kv_heads=1` is
multi-query; `kv_heads=heads` is plain multi-head.

In the code this is one asymmetry and one line. `k_proj` and `v_proj` output
`kv_heads × head_dim` channels where `q_proj` outputs `heads × head_dim`, and
`repeat_interleave` hands each KV head to its group just before scoring.
`state_bytes_per_token` reports the saving so profiling can compare it against
a constant-state mixer on the same axis.

## Order of operations

1. Project to q, k, v.
2. Split heads: `view` to `[b, s, h, d]`, *then* `transpose` to `[b, h, s, d]`.
3. `pos.rotate(q, k, pos_offset)` — queries and keys only.
4. Repeat KV heads to match query heads.
5. Score, scale by `sqrt(head_dim)`.
6. Mask the future to `-inf`.
7. Softmax in fp32.
8. Average the values, `transpose` back, flatten heads, output projection.

## Gotchas

**`view` before `transpose`, and the reverse on the way out.** The projection
lays channels out head by head, so `[b, s, h*d] → [b, s, h, d]` is a pure
reshape. Writing `view(b, h, s, d)` directly also produces the right shape and
runs without error — it just scrambles tokens and heads together. The output
side has the mirror trap: `transpose(1, 2)` must come before the flatten, and
the result needs `reshape` rather than `view` because the transpose left it
non-contiguous.

**Scale by `sqrt(head_dim)`, not `sqrt(d_model)`.** The scale exists so the dot
product of two random vectors has unit variance, which keeps softmax out of
saturation at init. The vectors being dotted are `head_dim` wide. Using
`d_model` over-shrinks the scores by `sqrt(heads)`, which flattens every
attention pattern toward uniform. The model still trains; it just starts with
attention that can barely discriminate. The SDPA oracle catches this, a shape
check does not.

**Mask with `-inf`, not a large negative number.** Softmax turns `-inf` into
exactly `0.0`, by construction, in any dtype. That exactness is what lets the
causality test demand bit-identical outputs with `torch.equal` rather than
`allclose` — a masked position contributes `0 × value`, which adds exactly
nothing. A finite fill like `-1e9` *also* gives an exact zero in fp32, but only
because `exp()` underflows; a milder constant such as `-100` leaves a denormal
weight of about `4e-45`, which is a real, if tiny, leak.

The difference that matters shows up when the mask is wrong. If every key in a
row is masked — an off-by-one that hides a token from itself does this to row 0
— `-inf` softmaxes to NaN and the first loss is NaN. `-1e9` softmaxes to a
*uniform* distribution over exactly the positions the token was forbidden to
see, future included, and training carries on. `-inf` is the fill that fails
loudly. A correct causal mask always leaves the diagonal, so the NaN case cannot
arise here; it can in sliding-window or document-masked variants.

**Softmax in fp32.** Scores in bf16 overflow `exp()` easily. Upcast for the
softmax, cast back after.

**Rotate before repeating KV heads.** Rotating `kv_heads` tensors is cheaper
than rotating `heads` copies, and the rotated, un-repeated keys are what the KV
cache will store in step 3.

**The grouping order is a convention weights depend on.** `repeat_interleave`
gives `[kv0, kv0, kv1, kv1]`; `repeat` gives `[kv0, kv1, kv0, kv1]`. Both
produce the right shape. Only the first matches how GQA checkpoints (and
SDPA's `enable_gqa`) assign query heads to groups. Same class of bug as the
RoPE layouts: silent, shape-correct, and wrong.

**Values are never rotated.** They are not scored against anything, so nothing
would cancel the turn; absolute position would ride into the output. With RoPE
and untouched values, the whole layer is shift invariant — asserted directly in
`tests/test_full_attention.py`.

**The mixer never asks what positional strategy it holds.** It calls
`pos.rotate` unconditionally. RoPE acts there; NoPE and the additive strategies
pass through. A NoPE attention layer in a hybrid is therefore a config edit.
The one constraint is that a rotary strategy must be built for this mixer's
`head_dim`, not `d_model`, and RoPE raises on the first call if it wasn't.

## What is tested

| Test | Catches |
|---|---|
| `test_mixers_equivalence.py::test_full_attention_matches_sdpa` | wrong scale, wrong mask, wrong grouping — MHA, GQA, MQA × NoPE, RoPE, at `atol=1e-6` |
| `test_causality.py::test_no_future_leakage_per_mixer` | any future leakage, bit-exact, by perturbation |
| `test_full_attention.py` | GQA shapes and cache bytes, group membership, first-token identity, shift invariance under RoPE, the positional seam, gradients, refusals |

All three were mutation-checked when written: an off-by-one mask, the
`sqrt(d_model)` scale, and `repeat` in place of `repeat_interleave` each fail
the suite, and the last fails *only* the grouped-query cases.

## Inference: the KV cache

Causality means a token's key and value never change once computed, so decode
keeps them (`infer/cache.py`) and each step computes one query, key and value.
The forward stays one code path: with no state the past is zero tokens long.

- **Keys are cached already rotated, and per KV head.** Rotation happens before
  `append`, because a token's position is fixed for good; and before
  `repeat_interleave`, so the cache is `kv_heads` wide, which is the whole of
  GQA's saving. Caching after the repeat would work and silently cost
  `heads / kv_heads` times the memory; the cache refuses a tensor of that shape.
- **The mask is a shifted triangle.** Query `i` sits at absolute position
  `past + i`, so it may see key `j <= past + i`: `tril(diagonal=past)` on an
  `[s, past + s]` matrix. Training is `past = 0`, the familiar square. Decoding
  one token is a single all-True row. The case in between — several tokens
  appended to a non-empty cache — is the one that is easy to forget and has its
  own test.
- **`is_causal=True` is only right for a square.** The fused kernel's flag means
  "lower triangle aligned top-left". Used while decoding one token it lets that
  token see only the *first* key. The fast tier therefore has three cases:
  `is_causal` for an empty cache, no mask at all for one token, and an explicit
  mask for a chunk onto a non-empty cache.
- **`pos_offset` must equal the cache length**, and the mixer checks. Decoding a
  lone token at the default `pos_offset=0` turns its query as if it opened the
  document; nothing crashes, the attention pattern is just wrong. For a
  full-attention cache slot `j` holds position `j`, so the mismatch is detectable
  and is raised.
- **A cache cannot slide.** Its keys were rotated at absolute positions, so
  dropping the oldest and carrying on would need every remaining key re-rotated.
  `generate` refuses a request that does not fit; `generate_uncached` slides,
  because it recomputes everything anyway.

**What the cache buys, measured.** On the 17M TinyStories model, one sequence:
on a CPU, 2.2x at 50 new tokens and 3.9x at 200, growing with length because
cache-free throughput falls while cached throughput is flat. On an Apple GPU it
is *slower* at 50 tokens and 1.4x at 200: with a model this small and a batch of
one, a decode step is bound by kernel-launch latency, not arithmetic, and the
cache only removes arithmetic. The cache is about FLOPs and memory traffic;
whether that is your bottleneck depends on model size and batch.
