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

## Not done yet

`state != None` raises. Incremental decode needs the KV cache from
`infer/cache.py`, which is ROADMAP step 3. The forward is already arranged for
it: keys are rotated at their true positions before being repeated, and
`pos_offset` is threaded through. What step 3 adds is appending to the cache
and widening the mask from `[s, s]` to `[s, past + s]`.
