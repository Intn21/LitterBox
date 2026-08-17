# Design notes

One page per technique: the math, the paper reference, and the gotchas.

These are written while implementing, not after. The value is concentrated in
the gotchas — the layout mismatch that cost an afternoon, the normalization that
matters more than expected, the place where the paper's notation and a working
implementation part ways.

Planned pages, roughly in implementation order:

| Note | Technique | Milestone |
|---|---|---|
| `full-attention.md` | MHA/GQA, causal masking, RoPE application | 1 |
| `rope-layouts.md` | Interleaved vs non-interleaved, and why it matters | 1 |
| `sliding-window.md` | Local attention, window/depth interaction in hybrids | 2 |
| `linear-attention.md` | Kernel feature maps, the parallel/recurrent duality | 3 |
| `deltanet.md` | The delta rule as an error-correcting state write | 3 |
| `gated-deltanet.md` | Data-dependent decay, state precision sensitivity | 3 |
| `dsa.md` | Lightning indexer, top-k selection, the two-stage recipe, and the deliberate RoPE-layout bug | 4 |
| `mla.md` | Latent KV compression and decoupled RoPE | 5 |

A note is worth writing the moment its mixer passes its first test, while the
confusion is still fresh enough to be worth recording.
