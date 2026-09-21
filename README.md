# LitterBox

A research sandbox for implementing, swapping, training, and benchmarking LLM
architecture techniques end-to-end, with a focus on long-context token mixers —
written out by hand, so that each technique is as legible as it is runnable.

> **Status: early, but the central idea works.** A decoder written from
> primitives — RoPE, grouped-query attention, SwiGLU, RMSNorm — is built from a
> YAML file, trains on TinyStories with a hand-written loop, and generates from a
> KV cache. **Swapping the token mixer is a config edit:** full attention,
> sliding-window attention, and a 3:1 hybrid of the two train and generate from
> one unchanged code path. 275 tests.
>
> Not there yet: the linear-attention family, DSA, MLA, and the eval harness are
> stubs, so nothing is *benchmarked*; and everything so far has run on a laptop,
> never on CUDA. [ROADMAP.md](ROADMAP.md) has the order things land in,
> [DEFERRED.md](DEFERRED.md) what is deliberately not built yet and why.

---

## Why

Everything the long-context field is currently arguing about — DSA, MLA, Gated
DeltaNet, KDA, sliding window, CSA/HCA — is a drop-in replacement for the same
thing: *how information moves between positions inside a layer.* If that
operation is a swappable module behind one interface, then a hybrid model
(say, 3 linear layers per 1 full-attention layer) is a config file rather than
a code fork, and comparing techniques is a sweep rather than a rewrite.

LitterBox is built to make that comparison cheap, and to make each technique
legible while doing it. The second half matters as much as the first: every piece
here exists twice, once as code meant to be *used* and once as a notebook meant
to be *understood*.

## Quickstart

```bash
git clone https://github.com/Intn21/LitterBox.git && cd LitterBox
uv sync --extra dev --extra demo        # or: pip install -e ".[dev,demo]"
pytest                                  # 275 pass; 5 xfail mark mixers not written yet
```

Pack a corpus, train a model, then swap its token mixer by naming a different file:

```bash
litterbox-pack configs/data/tinystories.yaml        # once: ~90 s, ~1 GB under data/

python examples/train_tinystories.py --model configs/models/tinystories-dense.yaml
python examples/train_tinystories.py --model configs/models/tinystories-swa.yaml
python examples/train_tinystories.py --model configs/models/tinystories-swa-hybrid.yaml
```

Those three models are identical except for `layer_pattern`. Nothing in the
script, the loop, or generation knows which one it was given. Each prints a
validation loss and a sampled story as it trains, resumes from its last
checkpoint if interrupted, and picks CUDA, then Apple's GPU, then CPU.

On an Apple M4, 300 steps (about 11 minutes) takes any of them from a loss of
10.8 to about 2.9 and to output like *"Once upon a time, there was a little boy
named Timmy. Timmy lived in a big house with his brother in the woods."*

## What's here

The goal is that no slot has exactly one implementation behind it. Where things
stand:

| Slot | Working | Stubbed or planned |
|---|---|---|
| Token mixer | full attention (MHA / GQA / MQA), sliding window — each in a reference and a fused tier | linear attention, DeltaNet, Gated DeltaNet, DSA, MLA |
| Positional | RoPE (both layouts), NoPE, learned table, sinusoidal (fixed or trainable) | YaRN |
| Channel mixer | SwiGLU | a second one (step 5) |
| Norm | RMSNorm | a second one (step 5) |
| Inference state | a growing KV cache, a rolling one that stops at the window | a fixed-size recurrent state (step 6) |
| Tokenizer | tiktoken, HuggingFace, SentencePiece, byte, char | a packaged BPE of your own |
| Data source | the Hub, local text, JSONL, anything `datasets` opens | — |
| Training | single-device loop: AdamW, warmup + cosine, bf16 autocast, accumulation, exact resume | multi-GPU adapter |
| Evaluation | — | NIAH, RULER, profiling, external-model loading |

## Design principles

**1. The token mixer is the unit of experimentation.** The core abstraction is a
registry of interchangeable mixer modules behind one `TokenMixer` interface,
composed per-layer via `layer_pattern` in a YAML config. Everything else is
injected too: a block receives its mixer, MLP and norms as instances and never
constructs or inspects them.

**2. Correctness oracles over trust.** Every implementation is checked against
something independent: full attention against PyTorch's fused kernel, RMSNorm
against `torch.nn.RMSNorm`, sliding window against a band mask built pair by
pair, cached generation against cache-free generation token for token, and the
DeltaNet family — when it lands — against `flash-linear-attention`. The
discrepancies are where the learning is.

**3. Naive first, fast second — but never naive-only.** Every mixer ships a
`reference/` implementation (pure PyTorch, readable, obviously correct, O(n²) is
fine) and optionally a `fast/` twin. The reference tier is both the pedagogical
product and the test oracle; the fast tier shares its parameters and checkpoint
keys, and `tier: fast` in a config swaps one for the other.

**4. Silent failures get a loud test.** The bugs that matter in this area do not
raise. Mixed RoPE layouts, a causal flag left on while decoding, a token decoded
without its position, a validation set that leaks: each produces the right
shapes, a falling loss, and fluent text. So the tests check *properties* —
perturb a token and nothing earlier may change, move a sequence and the scores
may not — and the code refuses what it can detect.

## The core interface

```python
class TokenMixer(nn.Module, ABC):
    """One layer's sequence-mixing operation. MLP/MoE lives outside."""

    @abstractmethod
    def forward(
        self,
        x: Tensor,                         # [batch, seq, d_model]
        state: MixerState | None = None,   # None during training (full-sequence)
        pos_offset: int = 0,               # absolute position of x[:, 0]
    ) -> tuple[Tensor, MixerState | None]: ...

    def init_state(self, batch_size, max_len, *, dtype, device) -> MixerState:
        """The empty inference state *this* mixer needs. Generation asks rather
        than guesses, and never learns whether it got a growing cache, a rolling
        one, or a fixed-size matrix."""

    @property
    @abstractmethod
    def state_bytes_per_token(self) -> float: ...

    def state_bytes(self, context_len: int) -> float:
        """Total state after n tokens. Unbounded by default; bounded mixers
        override it, and the gap between the curves is the case for them."""
```

Composition is config, never a code fork. This one trains today:

```yaml
# configs/models/tinystories-swa-hybrid.yaml
model:
  d_model: 256
  n_layers: 8
  vocab_size: 50304
  max_seq_len: 256
  layer_pattern:                 # tiles to n_layers: layers 3 and 7 are full attention
    - {mixer: sliding_window, heads: 8, kv_heads: 2, window: 64, pos: rope}
    - {mixer: sliding_window, heads: 8, kv_heads: 2, window: 64, pos: rope}
    - {mixer: sliding_window, heads: 8, kv_heads: 2, window: 64, pos: rope}
    - {mixer: full_attention, heads: 8, kv_heads: 2, pos: rope}
  mlp: {type: swiglu, hidden_mult: 4}
  norm: {type: rmsnorm, eps: 1.0e-5}
```

and this is where it is headed — the same shape, with the windowed layers
replaced by a mixer that has no KV cache at all
([`tiny-gdn-hybrid-3to1.yaml`](configs/models/tiny-gdn-hybrid-3to1.yaml) loads
today and builds once Gated DeltaNet is written):

```yaml
  layer_pattern:
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: gated_deltanet, heads: 12}
    - {mixer: full_attention, heads: 12, kv_heads: 2, pos: nope}
```

Three things are baked in deliberately.

- **Rotary and no-position strategies are per-layer**, because Kimi Linear puts
  NoPE on its full-attention layers and Llama 4's iRoPE interleaves RoPE and
  NoPE. Additive strategies — a learned or sinusoidal table — act once at the
  input, so they are set for the whole model (`pos: {type: learned}`); naming one
  per layer is an error that says where it goes.
- **A mixer's own arguments pass straight through** — `window`, an `indexer`
  block — and since no mixer accepts `**kwargs`, a misspelled key fails at build
  time, naming the layer.
- **The parallel/recurrent duality is a first-class test.** A full-sequence pass
  and step-by-step decode with state must agree, for every mixer, including a
  multi-token chunk appended to a non-empty cache. It is the most bug-prone
  property of the DeltaNet family and the cheapest test to keep green from day
  one.

## Learning the techniques

[`demo/`](demo/) holds a notebook per technique. Each states the idea in one
sentence, traces it by hand on an input small enough to check mentally, builds
the general version **from scratch without importing this library**, then breaks
it on purpose — preferring traps that produce plausible wrong output over ones
that raise — before showing the packaged equivalent. They run offline and end
with exercises, the last of which is always the next thing to build.

A reading order:

| Notebook | What you break on purpose |
|---|---|
| [tokenizers/bpe](demo/tokenizers/bpe.ipynb) | byte-pair encoding, trained and traced by hand |
| [data/preprocessing](demo/data/preprocessing.ipynb) | unshifted targets that make the loss look *better*; a `uint16` that wraps |
| [data/datasets](demo/data/datasets.ipynb) | three ways to leak a validation set, each reporting a flattering loss |
| [positional/sinusoidal](demo/positional/sinusoidal.ipynb) | a model with no positions that trains and stalls at exactly `ln(3)` |
| [positional/rope](demo/positional/rope.ipynb) | mixed layouts: nothing crashes, norms look healthy, scores stop meaning distance |
| [mixers/full_attention](demo/mixers/full_attention.ipynb) | no causal mask, and a loss seven times better than possible on random tokens |
| [infer/kv_cache](demo/infer/kv_cache.ipynb) | a decoded token that attends only to the first word — a valid distribution, no error |

[`docs/design-notes/`](docs/design-notes/) is the companion for someone reading
the source: one page per technique, centred on the gotchas that cost an
afternoon.

## Roadmap

Each step has an exit condition, and the next one doesn't start until it's met.
That rule is the guardrail against the framework becoming the project.

The goal, stated as something testable: **a decoder-only transformer where
swapping the token mixer is a config edit, and the new mixer trains and generates
without touching anything else.** Everything not on that path is deferred.

| Step | | Condition to move on |
|---|---|---|
| 0 ✅ | Tokenizer and data | A corpus packs; batches come out with targets shifted by one |
| 1 ✅ | Wiring, one implementation per slot | A forward pass produces a loss equal to `ln(vocab_size)` |
| 2 ✅ | It trains | Loss drops on TinyStories; output is vaguely English |
| 3 ✅ | It generates | Cached generation matches cache-free exactly |
| 4 ✅ | Prove the mixer seam | Dense, SWA, and a hybrid all train from unchanged code |
| 5 | Prove the other seams | Nothing has exactly one implementation behind it |
| 6 | Prove the state seam | Chunked parallel forward equals recurrent decode |

Step 6 is the one that decides whether the interface is real. Both mixers so far
are attention with a KV cache; a linear mixer has no cache at all, which is what
surfaces whether `MixerState` is secretly KV-shaped. After it, a new technique is
one file plus a config, and the research milestones begin: the eval harness, the
DeltaNet family, DSA, MLA.

[ROADMAP.md](ROADMAP.md) has the full version — what each step builds, the
attention pattern it introduces, and how each condition was met.
[DEFERRED.md](DEFERRED.md) is its counterpart: work deliberately *not* done yet,
what triggers picking it up, and what was paid up front to keep the door open.

## Results so far

One experiment, recorded in
[`experiments/2026-09-21-mixer-seam-trio/`](experiments/2026-09-21-mixer-seam-trio/notes.md):
three 18.5M-parameter models, identical but for `layer_pattern`.

| Model | Validation loss, 300 steps | KV cache after 200 tokens |
|---|---|---|
| full attention × 8 | 2.95 | 816 kB |
| sliding window (64) × 8 | 2.92 | 264 kB |
| 3:1 hybrid | 2.91 | 402 kB |

What it shows is structural: one code path, and a cache profile that follows the
config — in the hybrid only the two full layers keep growing. What it does not
show is anything about quality. It is one seed and 1% of an epoch, so the loss
ordering is noise until repeated; and whether a hybrid *recovers* what a window
*loses* needs a retrieval task past the window, which needs the eval harness.

## Repository layout

```
configs/          models, training, data, eval — composition happens here
src/litterbox/
  data/           tokenizers, lazy text sources, packing, data configs (litterbox-pack)
  model/          backbone, block, RMSNorm, SwiGLU, mixer registry, config -> model builder
  model/mixers/   reference/ (readable, the oracle) and fast/ (fused twins)
  positional/     rope, nope, learned, sinusoidal — behind one two-hook seam
  train/          the single-device loop
  infer/          KV caches, cached and cache-free generation
  eval/           harness, tasks, profiling — stubs for now
  utils/          the model config schema
tests/            the four test families, plus one file per component
demo/             one notebook per technique, built from scratch in the open
examples/         runnable scripts that use the library
experiments/      dated, self-contained records: config + hypothesis + result
docs/             the mixer guide and one design note per technique
```

Three structural choices worth calling out:

- **`experiments/` is version-controlled.** Research repos die when results live
  in scattered W&B runs. A dated folder holding config, notes, and `results.json`
  makes claims reproducible and gives contributors worked examples.
- **`positional/` is a top-level module, not something buried inside attention.**
  RoPE layout — interleaved vs non-interleaved — is exactly the bug class that
  bit DeepSeek's DSA indexer. Layouts are named, required, and tested, including
  a test that mixes them on purpose.
- **Data directories prove what they are.** A packed corpus carries the tokenizer
  that produced it and a fingerprint of its config. Loading it with a different
  tokenizer raises; re-running the pack command is free; changing the config and
  reusing the directory is refused rather than silently honoured.

## Practices

**Testing discipline.** Four test families are non-negotiable — oracle
equivalence, causality, state consistency, single-batch overfit. Cheap to write
early, brutal to retrofit.

**Tests must be able to fail.** A test that passes first time has not been shown
to test anything. Each component here was broken on purpose after its tests went
green — an off-by-one mask, a wrong scale, keys cached before rotation, a
dropped residual — to confirm the break was caught. One wasn't: a model missing
its residual connection still memorised a batch, which is why the block's
arithmetic has its own tests.

**Experiment hygiene.** Every training run that produces a claim gets an
`experiments/` folder with the exact config, a hypothesis written *before* the
run, and a conclusion written after.

**Compute honesty.** Everything through DSA is meant to be single-GPU feasible at
100M–350M scale on a 24–80GB card. *So far it has only run on an Apple-silicon
laptop.* The code is device-agnostic and was exercised end to end on a non-CPU
backend, but TF32, `torch.compile` and pinned-memory transfer are reasoned about
rather than tested, and the 100M configs have been built but never trained.

**Scope guardrail.** If a week goes by with no mixer implemented and no
experiment run — only infrastructure — stop and ship the nearest step in its
ugliest working form.

**Measure before optimising.** A process pool in the data pipeline turned out to
be a pessimisation. A KV cache turned out to be *slower* for short generations
from a small model on a GPU, where launch latency rather than arithmetic is the
cost. Neither would have been found by reasoning.

## What this is not

Not a serving engine. vLLM and SGLang solve a different problem and solve it
well; nothing here should be mistaken for production inference infrastructure.
Not a training framework either — multi-GPU lives behind a thin adapter in
[`train/distributed.py`](src/litterbox/train/distributed.py) and only shows up
once single-GPU work has outgrown itself. Not a reimplementation of RULER or
LongBench: the eval harness will wrap them. And not from-scratch for its own
sake: `nn.Linear` and `nn.Embedding` are used as-is, and the line is drawn at the
techniques the repo is about.

## Relationship to existing tools

| Tool | Role here |
|---|---|
| PyTorch SDPA / `nn.RMSNorm` | Oracles for the hand-written attention and norm; SDPA is also the fast tier |
| tiktoken / HF tokenizers / `datasets` | Real tokenizers and corpora, behind the repo's own interfaces |
| nanoGPT / litGPT | Style reference for readability; a loss-curve oracle for when claims are being made, not before |
| flash-linear-attention | Numerical oracle for the DeltaNet family; later, an optional fast-tier backend |
| torchtitan / accelerate | Swap-in backend behind `train/distributed.py` |
| RULER / LongBench v2 / HELMET | To be wrapped by the eval harness, not reimplemented |
| DeepSeek TileLang kernels | Readable reference for the DSA implementation |
| vLLM / SGLang | Out of scope — see [What this is not](#what-this-is-not) |

Version pinning is deliberate. Triton/PyTorch drift is the top source of phantom
bugs in this area (known `fla` kernel bugs on specific Triton versions, TileLang
FP8 not compiling on Ampere), so pins live in `pyproject.toml` and every
experiment note records its hardware and versions.

## Contributing

The repo is early and the interfaces may still move, but
[CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/adding-a-mixer.md](docs/adding-a-mixer.md) describe how a mixer gets added
and what tests it has to clear. If that guide is annoying to follow, the
interface is wrong, not the reader.

## License

[Apache 2.0](LICENSE).
