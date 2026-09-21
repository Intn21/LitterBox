# Demos

One notebook per technique, each building the thing from scratch in the open.

These are the explanation half of the repo. `src/litterbox/` holds implementations
written to be *used*; these notebooks hold the same ideas written to be
*understood* — traced by hand on tiny inputs first, then generalised, with the
traps hit deliberately rather than described.

A notebook here should be readable start to finish without running it, and
runnable start to finish without editing it.

## Running

```bash
uv sync --extra demo          # jupyter, regex, and the optional comparisons
jupyter lab demo/
```

Every notebook works with no optional dependencies and no network — cells that
would download a corpus or compare against a reference tokenizer degrade to a
printed note instead of failing.

## Contents

| Notebook | Technique | Status |
|---|---|---|
| [tokenizers/bpe.ipynb](tokenizers/bpe.ipynb) | Byte-pair encoding — train, encode, decode | ✅ |
| [data/preprocessing.ipynb](data/preprocessing.ipynb) | Text → sharded token files → batches, and the four silent failures along the way | ✅ |
| `tokenizers/unigram.ipynb` | Unigram LM — the opposite direction: prune a huge vocab with EM | planned |
| `mixers/full_attention.ipynb` | Causal attention, GQA, the KV cache | planned |
| `mixers/sliding_window.ipynb` | Local attention and what it costs you | planned |
| `mixers/linear_attention.ipynb` | Constant-size state, and the parallel/recurrent duality | planned |
| [positional/sinusoidal.ipynb](positional/sinusoidal.ipynb) | Position tables, learned and sinusoidal — why attention needs them, why a stack of waves doesn't repeat, and the distance trick RoPE is built on | ✅ |
| [positional/rope.ipynb](positional/rope.ipynb) | Rotary embeddings — position as a turn, the `rotate_half` trick traced by hand, and the interleaved/half layout bug that raises nothing | ✅ |

## Writing a new one

The shape that works, in order:

1. **The idea in one sentence.** If it needs a paragraph, you don't have it yet.
2. **Trace it by hand** on an input small enough to verify mentally — five words,
   eight tokens. Print the intermediate state after every step.
3. **Generalise** to the real function, then check it reproduces the hand trace.
4. **Break it on purpose.** Every technique has one or two traps that produce
   plausible-looking wrong output. Trigger them in a cell and show the error.
   A trap you've seen fire is worth ten you've read about.
5. **Scale up** to real data and measure the thing practitioners actually care
   about — bytes per token, or wall-clock, or memory.
6. **Exercises**, ending with the one that turns the notebook into a real
   implementation.

Notebooks build from scratch rather than importing `litterbox`. The point is to
put it in front of the reader; the packaged version is what you write
*afterwards*, informed by having done it once by hand. A closing section may then
show the packaged equivalent, once the reader has earned it — see the last
section of the preprocessing notebook.

Step 4 is the one that distinguishes these from documentation. Prefer traps that
produce *plausible* wrong output over ones that raise: a silently wrapped token
id or a target that leaks the answer teaches far more than an exception would,
because those are the ones that survive into real runs.
