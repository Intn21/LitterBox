# Roadmap

Seven milestones, eighteen weeks, then open-ended. The sequence runs measurement
first, correctness second, and techniques third — so that every claim later in
the list rests on something already verified.

> **The gating rule.** Each milestone ends at an exit criterion, and the next one
> does not start until that criterion is met. This is the guardrail against the
> framework becoming the project.

## Timeline

| Milestone | Weeks | Focus | Status |
|---|---|---|---|
| [M0](#m0--skeleton--eval-harness-against-external-models) | 1–2 | Skeleton + eval harness against external models | Scaffolding done |
| [M1](#m1--vanilla-transformer-trains-correctly) | 2–4 | Vanilla transformer trains correctly | Next |
| [M2](#m2--first-alternative-mixer--first-hybrid) | 4–6 | First alternative mixer + first hybrid | Queued |
| [M3](#m3--linear-attention-family) | 6–10 | Linear attention family | Queued |
| [M4](#m4--learned-sparsity--a-dsa-style-mixer) | 10–14 | Learned sparsity — a DSA-style mixer | Queued |
| [M5](#m5--mla--open-source-hardening) | 14–18 | MLA + open-source hardening | Queued |
| [M6+](#m6--frontier-territory) | ongoing | Frontier territory | Open-ended |

Note the shape of the calendar: M0–M2 are two weeks each, M3–M5 are four. More
than half the schedule is the linear-attention family and learned sparsity — the
first three milestones are the cheap part.

**Compute envelope.** Everything through M4 is single-GPU feasible at 100M–350M
scale on a 24–80GB card. Multi-GPU is an M5 concern and building it earlier is a
known way for the framework to become the project.

---

## M0 — Skeleton + eval harness against external models

**Weeks 1–2 · Scaffolding done, harness not started**

```
score
│•••              ← the product of this milestone: a length-degradation
│   •••             curve for a model somebody else already published
│      •            numbers for
│       ••
│         •
│          ••
│            ••
│              •●
└────────────────
  context length →
```

Build:

- [x] `TokenMixer` interface, mixer registry, repo scaffolding
- [x] Config system and schema validation
- [ ] Eval harness with NIAH — single-needle and multi-key
- [ ] Profiling: prefill/decode latency, KV bytes per token, peak memory
- [ ] `eval/external.py` wired to load HuggingFace models

Measurement comes first because it validates the instrument before any model code
exists. Every later claim in this roadmap depends on trusting the harness, and the
only way to earn that trust is to point it at a model somebody else has already
published numbers for.

> **Exit criterion.** The harness reproduces a published RULER/NIAH number for an
> existing small long-context model — Qwen2.5-7B-1M at 32K–128K, say — within a
> few points, and emits a length-degradation curve as both a plot and JSON.

---

## M1 — Vanilla transformer trains correctly

**Weeks 2–4 · Next**

```
█ · · · · · · ·    full attention
█ █ · · · · · ·    causal, dense
█ █ █ · · · · ·
█ █ █ █ · · · ·    every query sees every
█ █ █ █ █ · · ·    prior key — the shape
█ █ █ █ █ █ · ·    everything else is a
█ █ █ █ █ █ █ ·    departure from
█ █ █ █ █ █ █ █
```

Build:

- [ ] `full_attention` reference mixer — GQA plus RoPE
- [ ] Backbone: embeddings, blocks, norm, LM head
- [ ] Single-GPU training loop, written by hand rather than delegated
- [ ] Data pipeline on a small corpus — TinyStories to start, then FineWeb-Edu

This is the correctness anchor for everything afterwards. Every alternative mixer
gets compared against this one on short sequences where the two should agree
exactly.

> **Exit criterion — all four.**
>
> 1. The overfit-single-batch test passes.
> 2. A ~100M model trains to a loss curve matching nanoGPT on identical data and
>    hyperparameters, within noise.
> 3. The causality perturbation test passes.
> 4. Generation with a KV cache matches cache-free generation exactly.

---

## M2 — First alternative mixer + first hybrid

**Weeks 4–6 · Queued**

```
█ · · · · · · ·    sliding window
█ █ · · · · · ·    local band
█ █ █ · · · · ·
· █ █ █ · · · ·    everything outside the
· · █ █ █ · · ·    window is unreachable —
· · · █ █ █ · ·    which is exactly the
· · · · █ █ █ ·    failure the hybrid fixes
· · · · · █ █ █
```

Build:

- [ ] `sliding_window` mixer — the easiest non-trivial one
- [ ] The `layer_pattern` composition machinery
- [ ] A SWA/full hybrid config
- [ ] YaRN added to `positional/`

The payoff here is a signature rather than a score. A window-only model should
fail needle retrieval past its window; a hybrid with one full-attention layer in
four should recover it. Seeing that shape appear in your own plots is the proof
that harness, training, and composition all work together.

> **Exit criterion.** The hybrid trains stably, and the eval harness shows the
> expected signature: the SWA-only model fails NIAH beyond its window, and the
> hybrid recovers it.

**▸ Phase 0 complete.** Everything past this point is research inside a working
repo rather than work on the repo itself.

---

## M3 — Linear attention family

**Weeks 6–10 · Queued**

```
█ · · · · · · ·    linear / DeltaNet / GDN
▓ █ · · · · · ·    decaying recurrent state
▒ ▓ █ · · · · ·
░ ▒ ▓ █ · · · ·    no cache to grow: the past
░ ░ ▒ ▓ █ · · ·    is compressed into a
░ ░ ░ ▒ ▓ █ · ·    fixed-size matrix, and
░ ░ ░ ░ ▒ ▓ █ ·    distant tokens fade
░ ░ ░ ░ ░ ▒ ▓ █
```

Build:

- [ ] `linear_attention` — the educational baseline
- [ ] `deltanet` — the delta rule as an error-correcting state write
- [ ] `gated_deltanet` — plus data-dependent state decay
- [ ] Numerical validation against `flash-linear-attention`, as a test dependency
      and never a runtime one

Then the flagship experiment: a matched-budget ablation at 100M–350M — a
full-attention control against pure GDN and against hybrids at 7:1, 3:1, and 1:1 —
swept across context lengths on your own harness.

> **Exit criterion — all three.**
>
> 1. Numerical equivalence with `fla` within tolerance.
> 2. The parallel/recurrent consistency test passes — chunked forward equals
>    step-by-step decode.
> 3. The ablation reproduces the field's core finding — pure linear degrades on
>    multi-key retrieval, some hybrid ratio recovers it — written up as a note in
>    `experiments/`.

**Stretch, if the ablation lands early.** Quantize the recurrent state to fp8 and
measure the degradation, which tests MiniMax's precision-sensitivity claim
directly. And work out what prefix-cache reuse even means when the state is
recurrent rather than stored.

---

## M4 — Learned sparsity — a DSA-style mixer

**Weeks 10–14 · Queued**

```
█ · · · · · · ·    lightning indexer + top-k
█ █ · · · · · ·    learned sparsity
█ ░ █ · · · · ·
█ ░ █ █ · · · ·    column 0 is the attention
█ ░ ░ █ █ · · ·    sink; the diagonal is self;
█ ░ ░ ░ ░ █ · ·    the rest is whatever the
█ █ █ ░ ░ █ █ ·    indexer decides is worth
█ █ ░ █ ░ █ ░ █    reading
```

Build:

- [ ] Lightning indexer — a few ReLU heads — plus top-k selection over the KV cache
- [ ] Built as a mixer that wraps and extends full attention, not one that replaces it
- [ ] The two-stage recipe as a training-loop mode: freeze the model, fit the
      indexer with a KL loss against the dense attention distribution, then train
      jointly

**Deliberate exercise.** Implement, observe, and then fix an
interleaved/non-interleaved RoPE layout mismatch inside the indexer — the DeepSeek
bug — and write it up in `docs/design-notes/dsa.md`. It costs a day and it
inoculates you against the entire bug class.

> **Exit criterion.** A small dense-trained model, converted to sparse attention,
> retains its benchmark scores within a few points at the target length, with a
> measured reduction in attention FLOPs — and the indexer-recall diagnostic, the
> fraction of dense attention mass captured by the top-k, is implemented and
> reported.

---

## M5 — MLA + open-source hardening

**Weeks 14–18 · Queued**

```
▒ · · · · · · ·    multi-head latent attention
▒ ▒ · · · · · ·    low-rank KV
▒ ▒ ▒ · · · · ·
█ █ █ ▒ · · · ·    the same triangle carried
█ █ █ ▒ ▒ · · ·    at reduced rank — the cache
█ █ █ ▒ ▒ ▒ · ·    shrinks, the reachability
█ █ █ █ █ █ ▒ ·    does not
█ █ █ █ █ █ ▒ ▒
```

Build:

- [ ] `mla` mixer — latent KV compression with decoupled RoPE, much easier now
      that RoPE layouts are explicit and tested
- [ ] Optionally the first `fast/` Triton kernel, for whichever mixer profiling
      names as the bottleneck
- [ ] The open-source pass: `docs/adding-a-mixer.md`, CI running the suite, two or
      three polished example configs, a contribution guide

This is also where `train/distributed.py` stops being a stub, if and only if
single-GPU work has genuinely outgrown itself by now.

> **Exit criterion.** A stranger clones the repo, runs one command, and reproduces
> one experiment from `experiments/` end to end.

---

## M6+ — Frontier territory

**Ongoing · Open-ended**

```
█ · · · · · · ·    composite
█ █ · · · · · ·    local + sink + sparse
█ █ █ · · · · ·
█ ░ █ █ · · · ·    by now each of these is
█ ▓ ▓ █ █ · · ·    "add a mixer, run the
█ ▓ ▓ ░ █ █ · ·    standard ablation" — which
█ ░ ░ ░ ░ █ █ ·    is the entire point
█ ░ ░ ▓ ░ ░ █ █
```

- CSA/HCA-style sequence compression
- MSA-style block top-k
- KDA — per-channel gating over GDN, a natural increment once GDN exists
- NSA's three-branch design
- MoE in the MLP slot

This is also where genuinely novel small-scale studies live. Sequence compression
has the least independent replication in the literature, which makes it the most
valuable thing on this list to actually check.

---

## Cross-cutting practices

These start in week one, alongside M0. They are not a milestone and they never
become one. Each is cheap to adopt early and brutal to retrofit.

**Testing discipline.** Four test families, non-negotiable from M1: oracle
equivalence, causality, state consistency, single-batch overfit.

**Experiment hygiene.** Every run gets a folder with the exact config, a hypothesis
written *before* it, and a conclusion written after. Enforce on yourself what you
would want from contributors.

**Pin the stack.** Triton and PyTorch drift is the top source of phantom bugs here.
Pin versions, and record the GPU architecture in every experiment note.

**Compute honesty.** Everything through M4 fits on a single 24–80GB card at
100M–350M scale. Multi-GPU is an M5 concern.

**Scope guardrail.** If a week passes with no mixer implemented and no experiment
run — only infrastructure — stop and ship the nearest milestone in its ugliest
working form.

---

## What each external tool is for

Naming the role of each tool up front is what keeps the repo from slowly
reimplementing all of them.

| Tool | Role | Needed by |
|---|---|---|
| nanoGPT / litGPT | Loss-curve oracle; style reference for readability | M1 |
| flash-linear-attention | Numerical oracle for the DeltaNet family; later an optional fast-tier backend | M3 |
| RULER / LongBench v2 / HELMET | Wrapped by the eval harness, never reimplemented | M0 |
| DeepSeek TileLang kernels | Readable reference for the DSA implementation | M4 |
| torchtitan / accelerate | Swap-in backend behind `train/distributed.py` | M5+ |
| vLLM / SGLang | Out of scope — this is a research sandbox, not a serving engine | — |
