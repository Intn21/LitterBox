# LitterBox

A research sandbox for implementing, swapping, training, and benchmarking LLM
architecture techniques end-to-end, with a focus on long-context token mixers.

> **Status: early, but the central idea works.** A decoder written out by hand —
> RoPE, grouped-query attention, SwiGLU, RMSNorm — is built from a YAML file,
> trains on TinyStories with a hand-written loop, and generates from a KV cache.
> Swapping the token mixer is a config edit: full attention, sliding-window
> attention, and a 3:1 hybrid of the two train from one code path
> (`python examples/train_tinystories.py --model configs/models/<name>.yaml`).
> Linear-attention mixers, DSA, MLA and the eval harness are still stubs. See
> [ROADMAP.md](ROADMAP.md) for the order things
> land in, [DEFERRED.md](DEFERRED.md) for what is deliberately not being built
> yet, and [docs/adding-a-mixer.md](docs/adding-a-mixer.md) to fill one in.

---

## Why

Everything the long-context field is currently arguing about — DSA, MLA, Gated
DeltaNet, KDA, sliding window, CSA/HCA — is a drop-in replacement for the same
thing: *how information moves between positions inside a layer.* If that
operation is a swappable module behind one interface, then a hybrid model
(say, 3 linear layers per 1 full-attention layer) is a config file rather than
a code fork, and comparing techniques is a sweep rather than a rewrite.

LitterBox is built to make that comparison cheap, and to make each technique
legible while doing it.

## Design principles

**1. The token mixer is the unit of experimentation.** The core abstraction is a
registry of interchangeable mixer modules behind one `TokenMixer` interface,
composed per-layer via `layer_pattern` in a YAML config.

**2. Correctness oracles over trust.** Every mixer is validated against a
reference: full attention vs PyTorch SDPA, DeltaNet vs `flash-linear-attention`,
the training loss curve vs nanoGPT on identical data. The discrepancies are
where the learning is.

**3. Naive first, fast second — but never naive-only.** Every mixer ships a
`reference/` implementation (pure PyTorch, readable, obviously correct, O(n²) is
fine) and optionally a `fast/` one (Triton, fused, chunked). The reference tier
is both the pedagogical product and the test oracle, so any fast implementation
can be deferred without blocking research.

## What this is not

Not a serving engine. vLLM and SGLang solve a different problem and solve it
well; nothing here should be mistaken for production inference infrastructure.
Not a training framework either — multi-GPU lives behind a thin adapter in
[`train/distributed.py`](src/litterbox/train/distributed.py) and only shows up
once single-GPU work has outgrown itself. And not a reimplementation of RULER or
LongBench: the eval harness wraps them.

---

## Install

Requires Python ≥ 3.10 and a PyTorch build matching your CUDA version.

```bash
git clone https://github.com/Intn21/LitterBox.git
cd LitterBox
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest                       # stubs, so: expect xfails, not passes
```

Version pinning is deliberate. Triton/PyTorch drift is the top source of phantom
bugs in this area (known `fla` kernel bugs on specific Triton versions, TileLang
FP8 not compiling on Ampere), so pins live in `pyproject.toml` and every
experiment note records its GPU architecture.

## Repository layout

```
configs/          model / training / data / eval configs — composition happens here
src/litterbox/
  model/          backbone, block, mixer registry
  model/mixers/   reference/ (readable, correct) and fast/ (kernels, later)
  positional/     learned, sinusoidal, rope, yarn, nope — behind one two-hook seam
  train/          single-GPU loop, data pipeline, distributed adapter
  infer/          generation and the cache/state types
  eval/           harness, tasks (niah, ruler, multihop), profiling, external
  utils/          config schema, logging
tests/            the four non-negotiable test families
demo/             one notebook per technique, built from scratch in the open
experiments/      dated, self-contained records: config + hypothesis + result
docs/             contributor guide and one design note per technique
```

Three structural choices worth calling out:

- **`experiments/` is version-controlled.** Research repos die when results live
  in scattered W&B runs. A dated folder holding config, notes, and `results.json`
  makes claims reproducible and gives contributors worked examples.
- **`positional/` is a top-level module, not something buried inside attention.**
  RoPE layout — interleaved vs non-interleaved — is exactly the bug class that
  bit DeepSeek's DSA indexer. Naming and testing layouts explicitly is cheap
  insurance.
- **`eval/external.py` loads arbitrary HF models.** That's how the harness gets
  validated before any model code exists.

## Learning the techniques

[`demo/`](demo/) holds a notebook per technique, each building the thing from
scratch on inputs small enough to verify by hand, then scaling up to real data.
They deliberately trigger the traps — the ones that produce plausible-looking
wrong output rather than an error — so you've seen each failure fire before you
meet it in your own code.

Start with [byte-pair encoding](demo/tokenizers/bpe.ipynb).

## The core interface

```python
class TokenMixer(nn.Module, ABC):
    """One layer's sequence-mixing operation. MLP/MoE lives outside."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,  # [batch, seq, d_model]
        state: MixerState | None = None,  # None during training (full-sequence)
        pos_offset: int = 0,  # absolute position for RoPE at decode
    ) -> tuple[Tensor, MixerState | None]: ...

    @property
    @abstractmethod
    def state_bytes_per_token(self) -> float:
        """0 for constant-state mixers. Lets profiling treat mixers uniformly."""
```

Composition is config, never a code fork:

```yaml
# configs/models/tiny-gdn-hybrid-3to1.yaml
model:
  d_model: 768
  n_layers: 12
  layer_pattern:                  # tiles to n_layers
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: full_attention, heads: 12, kv_heads: 2, pos: nope}
  mlp: {type: swiglu, hidden_mult: 4}
```

Two subtleties are baked in deliberately. `pos` is **per-layer**, because Kimi
Linear puts NoPE on its full-attention layers and Llama 4's iRoPE interleaves
RoPE and NoPE — per-layer positional config expresses both for free. And the
**parallel/recurrent duality is a first-class test**: for linear mixers, the
chunked training-mode forward and the step-by-step inference recurrence must
agree exactly, which is the single most bug-prone property of the DeltaNet
family.

## Roadmap

Each milestone has an exit criterion, and the next one doesn't start until it's
met. That rule is the guardrail against the framework becoming the project.

The goal, stated as something testable: **a decoder-only transformer where
swapping the token mixer is a config edit, and the new mixer trains and generates
without touching anything else.** Everything not on that path is deferred.

| Step | | Condition to move on |
|---|---|---|
| 0 | Tokenizer and data ✅ | A corpus packs; batches come out with targets shifted by one |
| 1 | Wiring, one implementation per slot | A forward pass produces a loss equal to `ln(vocab_size)` |
| 2 | It trains | Loss drops on TinyStories; output is vaguely English |
| 3 | It generates | Cached generation matches cache-free exactly |
| 4 | Prove the mixer seam | Dense, SWA, and a hybrid all train from unchanged code |
| 5 | Prove the other seams | Nothing has exactly one implementation behind it |
| 6 | Prove the state seam | Chunked parallel forward equals recurrent decode |

Step 6 is the one that decides whether the interface is real — a linear mixer has
no KV cache at all, which is what surfaces whether `MixerState` is secretly
KV-shaped. After it, a new technique is one file plus a config, and the research
milestones (eval harness, the DeltaNet family, DSA, MLA) begin.

[ROADMAP.md](ROADMAP.md) has the full version — what gets built at each
milestone, the week ranges, and the attention pattern each one introduces.
[DEFERRED.md](DEFERRED.md) is its counterpart: work deliberately *not* done
yet, what triggers picking it up, and what was paid up front to keep the door
open.

## Practices

**Testing discipline.** Four test families are non-negotiable from step 1 —
oracle equivalence, causality, state consistency, single-batch overfit. Cheap to
write early, brutal to retrofit.

**Experiment hygiene.** Every training run gets an `experiments/` folder with the
exact config, a one-paragraph hypothesis written *before* the run, and a
conclusion written after.

**Compute honesty.** Everything through DSA is single-GPU feasible at 100M–350M
scale on a 24–80GB card. Multi-GPU lives behind `train/distributed.py` and
building it earlier is how the framework becomes the project.

**Scope guardrail.** If a week goes by with no mixer implemented and no
experiment run — only infrastructure — stop and ship the nearest step in its
ugliest working form.

**Measure before optimising.** Two bugs in the data pipeline were found by
benchmarking and would not have been found by reasoning: a process pool that was
a pessimisation, and a `.gitignore` pattern that silently excluded a package.

## Relationship to existing tools

| Tool | Role here |
|---|---|
| nanoGPT / litGPT | Loss-curve oracle for step 2; style reference for readability |
| flash-linear-attention | Numerical oracle for the DeltaNet family; later, an optional fast-tier backend |
| torchtitan / accelerate | Swap-in backend behind `train/distributed.py` |
| RULER / LongBench v2 / HELMET | Wrapped by the eval harness, not reimplemented |
| DeepSeek TileLang kernels | Readable reference for the DSA implementation |
| vLLM / SGLang | Out of scope — see [What this is not](#what-this-is-not) |

## Contributing

The repo is early and the interfaces may still move, but
[CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/adding-a-mixer.md](docs/adding-a-mixer.md) describe how a mixer gets added
and what tests it has to clear.

## License

[Apache 2.0](LICENSE).
