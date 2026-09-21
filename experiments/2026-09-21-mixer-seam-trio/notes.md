# The mixer seam: dense, sliding-window, and a 3:1 hybrid from one code path

- **Date:** 2026-09-21
- **Config:** `config.yaml` — one training config, three model configs
- **Hardware:** Apple M4, PyTorch MPS backend, unified memory. One run at a time.
- **Versions:** PyTorch 2.13.0, Python 3.12.13, macOS 26.6. No Triton, no CUDA.
- **Commit:** `ef07094`

## Hypothesis

This run exists to meet ROADMAP step 4's condition, which is about the *code*:
three model configs train from an unchanged code path, with only the YAML
differing. Falsified if any of them needs a change outside `configs/models/`.

The secondary prediction, about the *numbers*, was written before launch — in
the comment at the top of `configs/models/tinystories-swa.yaml`, not in this
file, which did not exist yet:

> window 64 against a 256-token context, so the window genuinely binds ... Through
> 8 layers influence still reaches 8 x 63 = 504 positions, further than the
> context — so expect a loss close to the dense model's. What a window costs is
> exact retrieval of something far back, and TinyStories, whose stories are ~200
> tokens, barely asks for that.

Falsified if the windowed model's validation loss is clearly worse than the dense
model's — by more than a few hundredths — at equal steps.

That comment was written before any run started and committed as `ef07094` while
the first run was inside its first hundred steps, before any number existed.

## Setup

Held constant: 18.52M parameters (identical in all three — a window adds none),
width 256, 8 layers, 8 query heads and 2 KV heads, SwiGLU, RMSNorm, RoPE, the
GPT-2 tokenizer, TinyStories, seed 1337, 300 steps of 16,384 tokens (4.9M
tokens), AdamW at 1e-3 with 30 warmup steps and cosine decay, bf16 autocast. The
same batches in the same order, since batches are a function of `(seed, step)`.

Varied: `layer_pattern` only.

| model | pattern |
|---|---|
| `dense` | full attention × 8 |
| `swa` | sliding window (64) × 8 |
| `swa-hybrid` | [window, window, window, full] × 2 |

"Matched budget" here means matched parameters, tokens and steps. It does *not*
mean matched compute per step in any useful sense: the reference tier computes a
full score matrix for windowed layers and masks it, so all three do the same
arithmetic.

## Result

See `results.json`.

| model | val @100 | val @200 | val @300 | train @300 | KV cache after 200 tokens |
|---|---|---|---|---|---|
| dense | 3.661 | 3.116 | 2.952 | 2.902 | 816 kB |
| swa | 3.630 | 3.074 | 2.915 | 2.861 | 264 kB |
| swa-hybrid | 3.621 | 3.072 | 2.911 | 2.858 | 402 kB |

All three trained from `examples/train_tinystories.py` with a different `--model`
and nothing else. Each writes recognisable TinyStories prose. For each trained
model, cached generation reproduces cache-free generation token for token over
200 tokens — for the hybrid that means three rolling caches and one full cache
per pattern repeat, which generation never distinguishes.

The cache column is the structural result. Per layer, a full cache holds 102 kB
at 200 tokens and a windowed one stops at 33 kB (64 tokens). The hybrid's profile
is `[33, 33, 33, 102, 33, 33, 33, 102]`: only its two full layers keep growing.

Not expected: the windowed models are slightly *ahead* of dense at every
evaluation, by 0.03–0.04.

## Conclusion

The primary hypothesis is supported. No file outside `configs/models/` changed
between the three runs. Step 4's condition is met.

The secondary prediction is supported in the sense that matters — the windowed
loss is not worse. It should not be read as "windows are better". This is **one
seed, 300 steps, 1% of an epoch**, and a 0.04 gap is within what a different seed
could produce. If the ordering is real, the likely reason is boring: a local
window is a good prior for children's stories, and early in training a good prior
beats flexibility. It says nothing about retrieval, which this corpus does not
exercise — the median story is 192 tokens.

Throughput differed by up to 9% between runs, which is machine contention (the
test suite ran during the dense run), not the mixers.

**What this does not show**, and the next experiment it implies: that a hybrid
*recovers* something a windowed model *loses*. That needs a task with a fact
placed beyond the window — needle-in-a-haystack — at a context several times the
window, which needs the eval harness (deferred) and more context than a laptop
trains comfortably. Until then the hybrid's case rests on its cache profile, not
on a quality result.

Also worth repeating with three seeds before anyone quotes the loss ordering.
