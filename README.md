# LitterBox

A research sandbox for implementing, swapping, training, and benchmarking LLM
architecture techniques end-to-end, with a focus on long-context token mixers.

> **Status: scaffolding.** The structure, interfaces, and roadmap are in place;
> the implementations are not. Every module below is a stub that raises
> `NotImplementedError`. See [Roadmap](#roadmap) for the order things land in,
> and [docs/adding-a-mixer.md](docs/adding-a-mixer.md) if you want to fill one in.

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
configs/          model / training / eval configs — composition happens here
src/litterbox/
  model/          backbone, block, mixer registry
  model/mixers/   reference/ (readable, correct) and fast/ (kernels, later)
  positional/     rope, yarn, nope — layouts are explicit, named, and tested
  train/          single-GPU loop, data pipeline, distributed adapter
  infer/          generation and the cache/state types
  eval/           harness, tasks (niah, ruler, multihop), profiling, external
  utils/          config schema, logging
tests/            the four non-negotiable test families
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

| # | Milestone | Exit criterion |
|---|---|---|
| 0 | Skeleton + eval harness on external models | Harness reproduces a published RULER/NIAH number for an existing small long-context model within a few points, emitting a degradation curve as plot + JSON |
| 1 | Vanilla transformer trains correctly | Overfit-single-batch passes; ~100M model matches a nanoGPT loss curve within noise; causality test passes; KV-cache generation matches cache-free generation exactly |
| 2 | First alternative mixer + first hybrid | Hybrid trains stably and the harness shows the expected signature: SWA-only fails NIAH past its window, hybrid recovers it |
| 3 | Linear attention family (DeltaNet, GDN) | Numerical equivalence with `fla`; parallel/recurrent consistency passes; matched-budget ablation across 7:1 / 3:1 / 1:1 hybrids, written up in `experiments/` |
| 4 | Learned sparsity: DSA-style mixer | Dense-trained model converted to sparse retains scores within a few points at target length, with measured attention-FLOPs reduction and an indexer-recall diagnostic |
| 5 | MLA + open-source hardening | A stranger clones the repo, runs one command, and reproduces one experiment end-to-end |
| 6+ | Frontier territory | CSA/HCA sequence compression, MSA block top-k, KDA, NSA, MoE in the MLP slot — by now each is "add a mixer, run the standard ablation" |

Milestone 0 comes first because it validates the measurement code before any
model code exists; every later claim depends on trusting the harness.

## Practices

**Testing discipline.** Four test families are non-negotiable from Milestone 1 —
oracle equivalence, causality, state consistency, single-batch overfit. Cheap to
write early, brutal to retrofit.

**Experiment hygiene.** Every training run gets an `experiments/` folder with the
exact config, a one-paragraph hypothesis written *before* the run, and a
conclusion written after.

**Compute honesty.** Everything through Milestone 4 is single-GPU feasible at
100M–350M scale on a 24–80GB card.

**Scope guardrail.** If a week goes by with no mixer implemented and no
experiment run — only infrastructure — stop and ship the nearest milestone in its
ugliest working form.

## Relationship to existing tools

| Tool | Role here |
|---|---|
| nanoGPT / litGPT | Loss-curve oracle for Milestone 1; style reference for readability |
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
