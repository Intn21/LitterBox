# Adding a mixer

Written for the repo's own author first — if this document is annoying to
follow, the interface is wrong, not the reader.

A mixer is one layer's sequence-mixing operation. Adding one should touch
exactly three places: a file in `reference/`, a design note, and tests. Nothing
in the backbone, the training loop, or the eval harness should need to learn
that your mixer exists.

## 1. Write the reference implementation

Create `src/litterbox/model/mixers/reference/<name>.py`:

```python
@register_mixer("my_mixer")
class MyMixer(TokenMixer):
    def __init__(self, d_model: int, heads: int, **kwargs) -> None: ...

    def forward(self, x, state=None, pos_offset=0): ...

    @property
    def state_bytes_per_token(self) -> float: ...
```

Rules for the reference tier:

- **Pure PyTorch, readable, obviously correct.** O(n²) is fine. This is the test
  oracle and the pedagogical product; a fused version comes later, in `fast/`,
  and is validated against this one.
- **Annotate shapes.** `# [b, h, n, d]` at each transform is the difference
  between a file people learn from and a file people skip.
- **Handle `state=None` and `state != None` in one code path** where you can.
  Two paths is two chances for them to disagree, and
  `test_state_consistency.py` exists precisely because they usually do.
- **`pos_offset` is not optional.** It is the absolute position of `x[:, 0]`, and
  ignoring it produces a model that trains fine and generates garbage.

## 2. Pick your positional handling explicitly

Import from `positional/` rather than inlining rotation math, and take the layout
(interleaved vs non-interleaved) from config. If your mixer has a second
attention-like path — an indexer, a compressed latent branch — that path needs
the *same* layout as the main one. A mismatch does not crash; it quietly costs
you accuracy, which is how it survived into a shipped model once already.

## 3. Make it configurable

Nothing to wire up: the registry resolves your name from `layer_pattern`. Add a
config under `configs/models/` demonstrating the mixer, ideally both pure and in
a hybrid.

## 4. Clear the four tests

| Test | What it catches |
|---|---|
| `test_mixers_equivalence.py` | Wrong math, against a trusted oracle |
| `test_causality.py` | Future leakage, via perturbation rather than mask inspection |
| `test_state_consistency.py` | Parallel forward and recurrent decode disagreeing |
| `test_overfit_single_batch.py` | Dead gradients and untrainable configs |

If your mixer has a constant-size state, expect to spend most of your debugging
time in the third one. That is normal and it is the point.

## 5. Write the design note

`docs/design-notes/<name>.md`, covering the math as you understand it, the paper
reference, and every gotcha you hit — especially the ones that cost you an
afternoon. The notes are much of what this repo is for; a mixer without one is
half a contribution.

## 6. Optionally, go fast

Once profiling says your mixer is the bottleneck, add
`src/litterbox/model/mixers/fast/<name>.py` mirroring the reference name. It gets
tested against the reference implementation, not against the paper.
