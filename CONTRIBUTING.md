# Contributing to LitterBox

The repo is early. The backbone, two attention mixers, training and generation
work; the linear-attention family, DSA, MLA and the eval harness are stubs with
their interfaces, configs and test slots already in place. `TokenMixer` has so
far only held mixers with a KV cache, so it may still move when the first
constant-state mixer lands. That makes it a good time to contribute a mixer and
a bad time to build infrastructure assuming the interface is frozen.

## Setup

```bash
uv sync --extra dev --extra eval    # or: pip install -e ".[dev,eval]"
pytest
ruff check . && ruff format --check .
```

## What contributions fit

**Good fits:** a new reference mixer, a design note for a technique that has one
implemented, a `fast/` kernel for a mixer that profiling shows is a bottleneck,
an eval task, or an experiment record that reproduces (or fails to reproduce) a
published result.

**Poor fits:** serving/deployment features, multi-GPU strategy beyond the thin
`train/distributed.py` adapter, or reimplementations of benchmark suites the
harness should be wrapping instead. See "What this is not" in the
[README](README.md#what-this-is-not).

## The bar for a new mixer

[docs/adding-a-mixer.md](docs/adding-a-mixer.md) has the walkthrough. In short, a
mixer is mergeable when:

1. It lives in `src/litterbox/model/mixers/reference/` in pure, readable
   PyTorch. O(n²) is acceptable in the reference tier — clarity beats speed.
2. It's registered with `@register_mixer("name")` and instantiable from a config
   with no code changes elsewhere.
3. It implements `init_state`, so generation can ask for its inference state
   without knowing what kind it is, and reports that state's cost honestly
   (`state_bytes_per_token`, plus `state_bytes` if the state is bounded). There
   is a test that compares the claim with measured bytes.
4. It passes the four test families:
   - **oracle equivalence** — matches a trusted reference numerically
   - **causality** — perturbing token *t* leaves outputs at positions `< t`
     bit-identical
   - **state consistency** — parallel/chunked forward equals step-by-step
     recurrent decode (the DeltaNet family's most bug-prone property)
   - **single-batch overfit** — a config using it can memorize one batch
5. Its constructor takes named arguments only — no `**kwargs` — so a misspelled
   config key raises instead of being swallowed.
6. Its tests have been shown to fail. After they go green, break the
   implementation in the ways this kind of code actually breaks — an off-by-one
   mask, a wrong scale, state updated before it is read — and confirm each break
   is caught. Say which you tried in the PR.
7. It ships a `docs/design-notes/<name>.md` covering the math, the paper
   reference, and the gotchas you hit. The gotchas are the valuable part.

A mixer without a design note is half a contribution — the notes are much of
what the repo is for.

## Experiment records

Every training run that produces a claim gets a folder under `experiments/`,
copied from `experiments/_template/`, containing the exact config, a hypothesis
written **before** the run, and a conclusion written after. Include the GPU
architecture and the resolved Triton/PyTorch versions; version drift in this
space produces bugs that look like research findings.

Negative results are welcome and get the same treatment as positive ones.

## Style

Ruff handles formatting and linting (`line-length = 100`). Beyond that: reference
implementations are read more than run, so favor explicit shapes in comments,
named intermediates over clever one-liners, and spelled-out einsum/rearrange
patterns over implicit broadcasting.

## Pull requests

Keep them scoped to one mixer, one task, or one fix. Note in the description
which tests you ran and on what hardware — CPU-only runs are fine to say so.
