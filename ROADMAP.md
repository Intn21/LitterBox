# Roadmap

**The goal, stated as something testable:**

> A decoder-only transformer you wrote, where swapping the token mixer is a
> config edit — and the new mixer trains and generates without touching anything
> else.

Everything that is not on the path to that sentence is deferred. See
[DEFERRED.md](DEFERRED.md) for what that means concretely and why.

Once the goal is reached, every technique in the field becomes "add a file, run
the standard comparison," which is the entire point of the repo. The research
milestones after it are listed below, but they are not the plan until then.

---

## Where things stand

| | |
|---|---|
| ✅ Working | tokenizers, data configs and packing, four positional strategies, full and sliding-window attention in two tiers each, RMSNorm, SwiGLU, the block, the backbone, model configs with layer-pattern tiling, the training loop, both KV caches and cached generation — 275 real tests |
| 🔨 Next | a second MLP and a second norm (step 5), then linear attention (step 6) |
| ⬜ Stubs | 15 files under `src/` are still entirely stubs |

Swapping the token mixer is now a config edit: a dense model, a sliding-window
model and a 3:1 hybrid train and generate from one code path. Both mixers are
attention with a KV cache, though, so the state interface has not yet met
something it was not shaped around.

---

## Getting to the goal

Each step ends at a condition. The next one does not start until it is met —
that rule is the guardrail against the framework becoming the project.

### Step 0 — Tokenizer and data ✅

**Done.** Text becomes token ids and lands on disk as something training can
read.

- [x] `Tokenizer` contract, with construction-time checks for the three ways
      backends silently disagree
- [x] tiktoken, HuggingFace, SentencePiece, byte, char
- [x] A comparison harness — bytes/token, throughput, round-trip, segmentation
- [x] Lazy sources; sharded `.bin` with a validated `meta.json`
- [x] `PackedDataset`: memmap reads, batches derived from `(seed, step)`

> **Condition met.** A corpus packs, a `PackedDataset` yields `[B, S]` batches
> whose targets are the inputs shifted by one, and loading shards packed by a
> different tokenizer raises.

### Step 1 — Wiring, one implementation per slot

```
█ · · · · · · ·    full attention
█ █ · · · · · ·    causal, dense
█ █ █ · · · · ·
█ █ █ █ · · · ·    every query sees every prior
█ █ █ █ █ · · ·    key — the shape everything
█ █ █ █ █ █ · ·    else is a departure from
█ █ █ █ █ █ █ ·
█ █ █ █ █ █ █ █
```

Config → builder → backbone → blocks, with everything injected and exactly one
thing behind each compartment.

- [x] `utils/config.py` — OmegaConf merges, pydantic validates, strictly. Landed
      with step 4, the first time a model could not be assembled from arguments
- [x] `positional/rope.py` — both layouts, named and tested, behind a
      positional seam (`embed`/`rotate` hooks) that also holds a GPT-2 style
      learned table, sinusoidal (fixed or learnable), and NoPE
- [x] `full_attention` — GQA plus an injected positional strategy; training path
      only, the KV cache is step 3
- [x] `block.py` — token mixer and channel mixer both injected, never constructed;
      pre-norm residuals, plus depth-scaled init via an opt-in flag
- [x] `transformer.py` — embeddings, the stack, final norm, LM head
- [x] SwiGLU, RMSNorm — `model/mlp.py` and `model/norm.py`, each with room for the
      second implementation step 5 asks for

The block must never learn what it is holding. The moment it grows an
`isinstance` check the compartment has stopped being one.

> **Condition.** A forward pass runs and produces a loss — and that loss equals
> `ln(vocab_size)`. Anything meaningfully below it means the targets are leaking;
> `demo/data/preprocessing.ipynb` has the assertion ready to lift.
>
> **Met, for a hand-assembled model** — `tests/test_backbone.py` asserts it
> through real blocks with shifted targets, `tests/test_causality.py` proves no
> position sees the future end to end, and
> `tests/test_overfit_single_batch.py` memorises a batch.

### Step 2 — It trains

- [x] Single-device loop, written by hand rather than delegated — AdamW with
      decay groups, warmup and cosine decay, bf16 autocast, gradient
      accumulation and clipping, JSONL logs, atomic checkpoints, exact resume
- [x] Consumes `PackedDataset`, prepared from a data config on launch
      (`configs/data/`, `litterbox-pack`)
- [x] Overfit-single-batch test passes, for a hand-assembled model
- [x] A fast attention tier (`full_attention_fast`), since the reference
      mixer's explicit score matrix cannot train the 100M config

> **Condition.** Loss drops on TinyStories and the output is vaguely English.
> *Not* matching a nanoGPT curve — that is a rigor step for when you are making
> claims, and it can eat a week.
>
> **Met.** `python examples/train_tinystories.py`: a 17M-parameter model, 600
> steps (10M tokens, 17 minutes on an Apple GPU), loss 10.8 → 2.48 with
> validation at 2.51, and it writes: *"Once upon a time, there was a little
> girl named Lily. She loved to play outside with her toys…"*
>
> Not yet run on CUDA — see *Untested on CUDA* in [DEFERRED.md](DEFERRED.md).

### Step 3 — It generates

- [x] KV cache in `infer/cache.py` — preallocated, written once per token, read as
      a view; sized by KV heads
- [x] Generation threading `MixerState`, advancing `pos_offset`: one prefill pass
      over the prompt, then one token of work per step
- [x] `init_state()` on the mixer, so generation never guesses what state is

Half of what distinguishes mixers is their *inference* path. A swap interface
that only works at training time is not the thing being built.

> **Condition.** Cached generation matches cache-free generation exactly, and
> perturbing token *t* leaves every output at position `< t` bit-identical.
>
> **Met.** `tests/test_generation.py` holds `generate` to `generate_uncached`
> token for token — greedy and sampled, MHA/GQA/MQA, both attention tiers, and
> with additive positions — and `tests/test_state_consistency.py` checks a
> parallel pass against step-by-step decode, including a multi-token chunk
> appended to a non-empty cache. On the trained TinyStories checkpoint the two
> generators agree exactly; the cache is 3.9x faster at 200 tokens on a CPU and
> the gap grows with length.

### Step 4 — Prove the mixer seam

```
█ · · · · · · ·    sliding window
█ █ · · · · · ·    local band
█ █ █ · · · · ·
· █ █ █ · · · ·    everything outside the window
· · █ █ █ · · ·    is unreachable — which is
· · · █ █ █ · ·    exactly the failure a hybrid
· · · · █ █ █ ·    is supposed to fix
· · · · · █ █ █
```

- [x] `sliding_window` — full attention with a different mask, literally: the
      mask is one overridable rule, and a windowed cache that stops growing
- [x] `layer_pattern` tiling, with `utils/config.py` and `model/build.py` turning a
      YAML file into a model
- [x] A SWA/full hybrid config

> **Condition.** `tiny-dense`, `tiny-swa`, and `tiny-swa-hybrid` all train from
> unchanged code — only the YAML differs.
>
> **Met**, at a size a laptop trains: the `tinystories-*` trio, 18.5M parameters
> each, identical but for `layer_pattern`. 300 steps from one script reach
> validation losses of 2.95 (dense), 2.92 (windowed) and 2.91 (hybrid), and each
> trained model generates from its caches exactly what it generates without them
> — the hybrid from three rolling caches and one full cache per repeat. After 200
> tokens those caches hold 816, 264 and 402 kB. Written up in
> `experiments/2026-09-21-mixer-seam-trio/`.
>
> One seed and 1% of an epoch: the loss ordering is not a finding. And that a
> hybrid *recovers* what a window *loses* is not shown here — it needs a
> retrieval task past the window, which needs the eval harness.

### Step 5 — Prove the other seams

- [ ] A second channel mixer (plain MLP or GeGLU)
- [ ] A second norm, or a NoPE layer inside a hybrid

Cheap — a day's work. Without it, those compartments are still hypotheses. A
seam with one implementation behind it has not been tested as a seam.

> **Condition.** Nothing has exactly one implementation behind it.

### Step 6 — Prove the state seam

```
█ · · · · · · ·    linear attention
▓ █ · · · · · ·    constant-size state
▒ ▓ █ · · · · ·
░ ▒ ▓ █ · · · ·    no cache to grow: the past is
░ ░ ▒ ▓ █ · · ·    compressed into a fixed matrix,
░ ░ ░ ▒ ▓ █ · ·    and distant tokens fade
░ ░ ░ ░ ▒ ▓ █ ·
░ ░ ░ ░ ░ ▒ ▓ █
```

- [ ] `linear_attention` — a feature map and a running sum, no delta rule, no
      gating

This is the step that decides whether the interface is real. Sliding window
reuses the same KV cache, the same shapes, the same decode path — swap it in and
everything works because nothing structural changed. A linear mixer has **no KV
cache at all**, which is what surfaces whether `MixerState` is secretly
KV-shaped or the decode loop assumes a growing cache.

Most of what is arriving in the field is in that family, so an interface that
cannot hold a constant-state mixer fails on the majority of what you want to
test — and you would find out four implementations in.

> **Condition.** The chunked parallel forward equals step-by-step recurrent
> decode. That is the most bug-prone property in this whole repo, and
> `tests/test_state_consistency.py` exists for it.

---

## 🎯 Goal reached

At this point every seam has two implementations, and a new technique is one
file plus a config. Everything below is research *using* the repo rather than
work *on* it.

---

## After the goal

Not the plan until the goal is met. Ordered by what unblocks what.

### Eval harness

NIAH single and multi-key, RULER wrapping, profiling, and `eval/external.py` to
load arbitrary HF models.

Deliberately **not** first, despite being the original M0. The four test
families validate an *implementation* without it — oracle equivalence,
causality, state consistency, single-batch overfit are all `torch.allclose` and
property checks. A harness is required for *claims*, and the first claim is the
hybrid ablation.

**Exit criterion.** Reproduces a published RULER/NIAH number for an existing
small long-context model within a few points, emitting a degradation curve as
plot and JSON.

### Linear attention family

`deltanet`, `gated_deltanet`, validated numerically against
`flash-linear-attention` as a test dependency and never a runtime one.

Then the flagship experiment: a matched-budget ablation at 100M–350M — a
full-attention control against pure GDN and hybrids at 7:1, 3:1, and 1:1 —
swept across context lengths.

**Exit criterion.** Equivalence with `fla` within tolerance; the ablation
reproduces the field's core finding (pure linear degrades on multi-key
retrieval, some ratio recovers it), written up in `experiments/`.

### Learned sparsity — DSA

```
█ · · · · · · ·    lightning indexer + top-k
█ █ · · · · · ·
█ ░ █ · · · · ·    column 0 is the attention sink;
█ ░ █ █ · · · ·    the diagonal is self; the rest
█ ░ ░ █ █ · · ·    is whatever the indexer decides
█ ░ ░ ░ ░ █ · ·    is worth reading
█ █ █ ░ ░ █ █ ·
█ █ ░ █ ░ █ ░ █
```

A few ReLU indexer heads plus top-k selection over the KV cache, wrapping full
attention rather than replacing it. Two-stage training: freeze the model, fit
the indexer with a KL loss against dense attention, then train jointly.

Includes implementing, observing, and fixing an interleaved/non-interleaved RoPE
mismatch in the indexer — the DeepSeek bug — as a documented exercise.

**Exit criterion.** A dense-trained model converted to sparse holds its scores
within a few points at target length, with measured attention-FLOPs reduction
and an indexer-recall diagnostic.

### MLA and open-source hardening

```
▒ · · · · · · ·    multi-head latent attention
▒ ▒ · · · · · ·    low-rank KV
▒ ▒ ▒ · · · · ·
█ █ █ ▒ · · · ·    the same triangle at reduced
█ █ █ ▒ ▒ · · ·    rank — the cache shrinks, the
█ █ █ ▒ ▒ ▒ · ·    reachability does not
█ █ █ █ █ █ ▒ ·
█ █ █ █ █ █ ▒ ▒
```

Latent KV compression with decoupled RoPE, much easier once RoPE layouts are
explicit and tested. Optionally the first `fast/` Triton kernel, for whichever
mixer profiling names as the bottleneck. Then CI, polished configs, and a
contribution guide.

**Exit criterion.** A stranger clones the repo, runs one command, and reproduces
one experiment from `experiments/` end to end.

### Frontier

CSA/HCA sequence compression, MSA block top-k, KDA, NSA's three-branch design,
MoE in the MLP slot. By now each is "add a mixer, run the standard ablation."

Sequence compression has the least independent replication in the literature,
which makes it the most valuable thing here to actually check.

---

## Cross-cutting practices

These start now, not at a step. Cheap to adopt early, brutal to retrofit.

**Testing discipline.** Four families, non-negotiable from step 1: oracle
equivalence, causality, state consistency, single-batch overfit.

**Experiment hygiene.** Every run gets a folder with the exact config, a
hypothesis written *before* it, and a conclusion written after.

**Pin the stack.** Triton and PyTorch drift is the top source of phantom bugs
here. Pin versions, record the GPU architecture in every experiment note.

**Compute honesty.** Everything through DSA is single-GPU feasible at 100M–350M
on a 24–80GB card. Multi-GPU is a `train/distributed.py` concern and building it
earlier is how the framework becomes the project.

**Scope guardrail.** If a week passes with no mixer implemented and no
experiment run — only infrastructure — stop and ship the nearest step in its
ugliest working form.

**Measure before optimising.** Two bugs in the data pipeline were found by
benchmarking and would not have been found by reasoning: a process pool that was
a pessimisation, and a `.gitignore` pattern that silently excluded a whole
package. Numbers first.

---

## What each external tool is for

Naming the role up front is what stops the repo slowly reimplementing all of them.

| Tool | Role | Needed by |
|---|---|---|
| tiktoken / HF tokenizers | Real tokenizers to compare your own against | step 0 ✅ |
| nanoGPT / litGPT | Loss-curve oracle; style reference for readability | step 2 |
| RULER / LongBench v2 / HELMET | Wrapped by the eval harness, never reimplemented | eval harness |
| flash-linear-attention | Numerical oracle for the DeltaNet family | linear family |
| DeepSeek TileLang kernels | Readable reference for the DSA implementation | DSA |
| torchtitan / accelerate | Swap-in backend behind `train/distributed.py` | multi-GPU |
| vLLM / SGLang | Out of scope — a research sandbox, not a serving engine | — |
