# Tiny Shakespeare, character level: nanoGPT's recipe without dropout

- **Date:** 2026-09-23
- **Config:** `config.yaml`
- **Hardware:** Apple M4, PyTorch MPS backend, 16 GB unified memory. Wall clock 78 minutes.
- **Versions:** PyTorch 2.13.0, Python 3.12.13, macOS 26.6. No CUDA.
- **Commit:** `dc1a8c4`

## Hypothesis

*Written before the run.* This is nanoGPT's `train_shakespeare_char.py` recipe —
6 layers, width 384, 6 heads, 256-token context, batch 64, lr 1e-3, 5,000
steps — on this repo's parts (RoPE, SwiGLU, RMSNorm) instead of GPT-2's. nanoGPT
reports a validation loss near 1.47 around step 2,000 and says it overfits after
that. Expectation: a similar curve, bottoming out somewhere near 1.5 around step
2,000, and the first outside number this repo can be compared against.

Falsified if the validation loss never gets near 1.5, or if the curve's shape
differs materially — a much earlier or later minimum.

The comment at the top of `configs/training/shakespeare-char.yaml`, committed
before the run, states the same expectation.

## Setup

One run, seed 1337. 16,384 tokens per step over a 1,003,854-token training
split: one epoch every 61 steps, 82 epochs in total. Validation is the last 10%
of the file by position, 111,540 characters, scored on 20 fixed batches.

What differs from nanoGPT, deliberately: the block (RoPE, SwiGLU, RMSNorm, no
biases) and — not deliberately, because the model has no such option — **no
dropout**. nanoGPT's recipe sets `dropout = 0.2` for this dataset.

## Result

See `results.json`.

| step | train | val | |
|---|---|---|---|
| 250 | 1.345 | 1.580 | |
| 500 | 1.134 | **1.522** | minimum |
| 750 | 0.967 | 1.600 | |
| 1,000 | 0.747 | 1.840 | best checkpoint that survived |
| 2,000 | 0.230 | 3.061 | where nanoGPT's minimum is |
| 5,000 | 0.075 | 4.409 | final |

Validation bottomed at step 500 — eight epochs — and rose for the remaining
4,500 steps while training loss fell to 0.075. Throughput 17,400 tokens/s.

The overfitting is visible in the text. Sampled at temperature 0.8, of lines
20+ characters long:

| checkpoint | lines found verbatim in the training text |
|---|---|
| step 1,000 | 0 of 38 |
| step 5,000 | 30 of 37 |

The step-5,000 model, prompted with `ROMEO:`, recites Romeo's speech at the tomb
("Thou detestable maw, thou womb of death, / Gorged with the dearest morsel of the
earth"). It has memorised the plays. The step-1,000 model writes new lines in the
right register, with invented but plausible names and metre.

Not expected: the minimum arrived at step 500, four times earlier than
nanoGPT's, and the best checkpoint was lost. Checkpoints are written every 500
steps and only the latest is kept, so by the time the trend was clear (step
1,000) the step-500 file had been overwritten. A copy of step 1,000 was taken by
hand before step 1,500 replaced it.

## Conclusion

The hypothesis is half supported. The model reaches the neighbourhood of
nanoGPT's number — 1.52 against their 1.47 — so the loop, the model and the data
path are doing what they should, and this is the first outside comparison the
repo has passed. The curve's shape is falsified: the minimum is at step 500, not
2,000, and the rise afterwards is far steeper.

The cause is almost certainly dropout, which nanoGPT uses at 0.2 on this dataset
and this model does not have. With 10.7M parameters and 1M tokens of training
text, the model can hold the corpus outright, and nothing stops it; nanoGPT's
0.2 dropout is what buys it the extra 1,500 steps before the same thing happens.
Weight decay at 0.1 is not enough on its own. The remaining 0.05 between 1.52
and 1.47 may be dropout too, or the architectural differences, or seed; one run
cannot say.

Two things to do, recorded in `DEFERRED.md`:

- **Dropout**, as an option on the block's two residual branches, so the recipe
  can be matched properly. Then rerun and expect the minimum to move to ~2,000.
- **Keep the best checkpoint, not only the latest.** A run whose validation loss
  rises should still be able to hand back the model from before it did. This is
  the second time in the repo that "latest" was the wrong thing to keep.

The result the run was launched for — a checkpoint to generate from — is
`runs/shakespeare-char/latest.pt`, which is the step-1,000 copy. The step-5,000
file is kept alongside as `step-5000.pt` for the contrast above.

What this does not show: anything about which architecture is better. It is one
seed and a different model from nanoGPT's in four ways at once.
