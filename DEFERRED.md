# Deferred

Work deliberately not done yet, and why. Distinct from [ROADMAP.md](ROADMAP.md),
which is the sequence of things we *intend* to build — this is the record of
choices to **not** build something now, so that a decision made once doesn't get
relitigated every time someone notices the gap.

Each entry says what triggers picking it up, how expensive it looks to add
later, and — where it applies — what small cost was paid up front to keep the
door open. That last column is the one that matters. A deferral is cheap when
you've left a hook and expensive when you have to unpick a decision.

## Keeping it honest

This file is only worth having if it stays current. A stale list of deferrals is
worse than none: it makes work look pending that already landed, and hides
decisions taken since.

So — **delete an entry when the work ships**, naming the commit in the message
rather than leaving a tombstone here. And **add one whenever you consciously
choose not to build something**, at the moment you make the call, while the
reasoning is still in your head. An entry written weeks later is a guess about
why past-you did something.

Entries are also fair to *revise*. If a trigger fires and you decide to keep
deferring, say so and say why — that is a different decision from the original
one and worth recording as such.

---

## Data pipeline

### Length curriculum

`PackedDataset` takes a fixed `seq_len` at construction. Every batch for the
whole run is the same shape.

The alternative grows the length during training — most of the run short and
cheap, then a brief phase at full length. `configs/training/longctx-extension.yaml`
already describes the recipe: 2048 → 8192 → 32768, with `batch_size` moving
inversely so tokens-per-step stays at 65,536.

**Why deferred.** The dataset side is small. The *training loop* is what has to
become stage-aware: track the current stage, resize batches, switch on RoPE
scaling, and size the model's `max_seq_len` for the largest stage. No training
loop exists yet, so building the data side against it would be guessing.

**Insurance paid.** `get_batch(batch_size, step, ...)` already takes `step`. That
is the only reason a schedule can hook in without touching a single caller.

**Trigger.** The first time you want to train past the length you pretrained at
— which in practice means after the goal in [ROADMAP.md](ROADMAP.md) is reached.

**Cost later.** Low. `n_valid_positions` is derived from `seq_len` at
construction, so either recompute it on change or accept `seq_len` per call.
Contained inside `PackedDataset`.

### Intra-document attention masking

Documents are concatenated with EOS between them, and a sampled window can span
a boundary — so a sequence sometimes attends across two unrelated documents.
Standard for pretraining, and generally accepted as fine.

**Why deferred.** Needs the model to accept a per-token document id and build a
block-diagonal mask. No model exists yet.

**Insurance paid.** `pack()` writes `{split}_docs.bin` — the global token index
where each document starts — by default. It costs ~8 bytes per document and
**cannot be recovered later without repacking the entire corpus**, which is the
whole reason it's written now despite nothing reading it.

**Trigger.** Evidence that cross-document contamination is costing you, or an
SFT path (where it stops being optional).

**Cost later.** Low *given the offsets*. Prohibitive without them.

### Best-fit packing

Documents are concatenated in arrival order. Bin-packing them to reduce how
often a document gets truncated mid-sequence is a real, published improvement.

**Trigger.** After the basic pipeline has actually trained something.
**Cost later.** Low — it's a reordering step before the shard writer.

### Multi-node and object storage

Shards are local files read with `np.memmap`. Streaming from S3/GCS with elastic
resharding and mid-epoch resume is what Mosaic Streaming solves.

**Why deferred.** It solves a problem single-node training does not have. At
measured cost — 37 µs to read a `B=32, S=1024` batch against tens of
milliseconds for a forward+backward — reading is under 1% of a step.

**Trigger.** Multi-node training. Same bracket as `train/distributed.py`.
**Cost later.** Moderate, and self-contained behind `PackedDataset`'s interface.

---

## Tokenizer

### A packaged BPE implementation

There is no `bpe.py`. The working byte-level BPE lives only in
`demo/tokenizers/bpe.ipynb`.

**Why deferred.** Implementing it is the point of the exercise — the notebook
teaches it, and exercise 5 is the on-ramp to a real class. Any algorithm becomes
first-class by decorating it with `@register_tokenizer`; no special slot needed.

**Trigger.** Whenever you want to train on a tokenizer you built.
**Cost later.** None structural — it's an additional registered class.

### Tokenizer save/load

Nothing persists a trained-in-memory tokenizer.

**Why it matters more than it looks.** `pack()`'s process pool rebuilds the
tokenizer inside each worker *from a config*. A tokenizer that exists only as
in-memory state can't be reconstructed that way, so it silently falls back to
serial encoding. Fine today — serial uses `encode_batch` and is fast — but it
does mean a custom tokenizer can't currently use workers.

**Trigger.** The first tokenizer you train and want to reuse.
**Cost later.** Low. `save()`/`load()` on `TrainableTokenizer`, plus a `type`
that builds from a file path.

### The ranked (fast) encoder

`encode` replays every merge in order. The standard optimisation looks at the
pairs actually present and repeatedly applies the lowest-ranked one — identical
output, work proportional to merges that fire rather than to vocabulary size.

**Trigger.** When your own BPE feels slow.
**Cost later.** None. Keep the naive version as the oracle, same reference/fast
split the mixers use.

### Streaming decode

`decode()` joins all bytes and decodes once, which is correct. Printing tokens
*as they generate* needs a byte buffer that only flushes complete UTF-8
sequences, or a multi-byte character split across two tokens raises.

**Trigger.** The first interactive generation demo.
**Cost later.** Trivial, and local to the caller.

### Chat templates

Nothing turns a list of `{role, content}` messages into the single string a
model actually sees. HuggingFace's answer is a good one and the plan is to adopt
it as-is: a **Jinja2 template stored as one string** in the tokenizer's config
(`chat_template` in `tokenizer_config.json`), rendered by
`apply_chat_template`. The conversation format becomes data, so swapping ChatML
for the Llama format is a config edit — the same idea as a mixer being a name
in a `layer_pattern`.

A template does not make a model able to chat. It is a contract: the model
learns the format by being fine-tuned on text rendered through it, and
inference must render with the identical template. That is why it belongs with
the tokenizer rather than with generation — same reason a RoPE layout belongs
with the weights.

**Decided now, so it is not relitigated later.**

- Jinja2, field-compatible with HuggingFace, so `HFTokenizer` can use whatever
  template a Hub model ships with and ours can be loaded elsewhere.
- Rendered in Jinja's `ImmutableSandboxedEnvironment`. A template is code, and
  one downloaded from the Hub is untrusted code.
- Role markers such as `<|im_start|>` must each be a single token, registered as
  special tokens — otherwise the model reads them as ordinary text.
- Assistant spans marked in the template (HF's `{% generation %}` block) so the
  SFT loss mask falls out of rendering instead of being reverse-engineered from
  the string.

**Why deferred.** Pretraining has no roles; TinyStories is plain text and there
is nothing to template. The only consumer is the SFT data path below, which is
itself deferred. A render function with nothing downstream would go untested
against real use.

**Insurance paid.** The model configs pad `vocab_size` to 50304 for kernel
alignment, and GPT-2's tokenizer uses 50257 ids. That leaves **47 embedding rows
already allocated and unused** — enough for every role marker a chat format
needs, with no embedding resize and no change to a pretrained checkpoint's
shapes. They never receive gradient during pretraining (absent tokens get
exactly zero), so they arrive at fine-tuning at their init values.

**Trigger.** The first instruction-tuning run, or wanting to chat with a Hub
model through `eval/external.py`.

**Cost later.** Low — about fifty lines plus tests, and `jinja2` is already a
transitive dependency of `transformers`. The one fiddly part is the assistant
mask. Build it together with the SFT data path, not before.

---

## Model

### What is still a stub

15 files under `src/litterbox/` are still entirely stubs: five of the seven
mixers (linear attention, DeltaNet, Gated DeltaNet, DSA, MLA), logging backends,
YaRN, the training-side data and distributed adapters, and the whole eval
harness.

What stands: the positional strategies; full attention and sliding-window
attention, each in two tiers with its inference path; RMSNorm, SwiGLU, the block
and the backbone; model configs with layer-pattern tiling; data configs and
packing; the training loop; both KV caches; and cached generation. A dense model,
a sliding-window model and a 3:1 hybrid of the two train from one code path with
only the YAML differing.

An earlier full implementation was written and then **deliberately reverted**
(`31ad17a`, reverted by `ff7eb66`) — building it is the practice this project
exists for.

**Trigger.** Now. Step 5 in [ROADMAP.md](ROADMAP.md) — a second MLP and a second
norm, so no seam has one implementation behind it — then step 6, linear
attention, whose state is not a cache at all.

### `positions` tensor instead of `pos_offset: int`

`TokenMixer.forward` takes `pos_offset: int`, which assumes positions are
contiguous from an offset. True for every case in this project today.

**What breaks it.** Packed SFT, where several examples share one sequence and
each must restart at position 0. An int cannot express that.

**Trigger.** An SFT path, or packed variable-length training.
**Cost later.** Mechanical but wide — the signature change touches every mixer.

---

## Training

### Untested on CUDA

The loop was written for CUDA and has only ever run on Apple's GPU and the CPU,
because that is the hardware it was written on. Three paths are therefore
reasoned about rather than exercised: TF32 matmuls
(`set_float32_matmul_precision("high")`), `training.compile`, and
`PackedDataset`'s pinned-memory transfer.

**Insurance paid.** Nothing names a device: tensors follow the model and the
batch, and the model was run end to end on a non-CPU backend, which fails the
same way CUDA does when a tensor is created on the wrong device. Checkpoints
load through the CPU, so a run started on one kind of machine opens on another.
RoPE's tables are pinned to fp32 through model-wide casts.

**Trigger.** The first session on a CUDA machine. Run the suite, then
`examples/train_tinystories.py training.compile=true` for a few hundred steps,
and delete this entry.

### Windowed attention that is cheap to *train*

Sliding-window attention saves memory at inference, where its cache stops
growing. It saves nothing in training: both tiers still compute a full `[s, s]`
score matrix and mask most of it away, so compute and activation memory are
those of full attention.

**Why deferred.** Skipping the masked blocks is a kernel — FlexAttention's block
masks, or FlashAttention's native `window_size` — and the reference tier's job is
to be obviously correct. At TinyStories scale (256 tokens, window 64) there is
nothing to save.

**Trigger.** Training a windowed or hybrid model at a context where attention
dominates the step — a few thousand tokens at 100M parameters. The 100M SWA
configs are written for 8,192.
**Cost later.** Low to moderate, and contained in `mixers/fast/sliding_window.py`.
The reference mixer is the oracle it gets tested against.

### A loss that does not materialise the logits

Logits are `batch x seq x vocab` floats. With GPT-2's vocabulary that is 3.3 GB
in fp32 at a micro-batch of 64 x 256 — before its gradient — and it, not the
model, is what sets the largest micro-batch that fits. A laptop GPU tops out
around 32 x 256 on a 17M-parameter model for this reason alone.

**Why deferred.** Gradient accumulation already gives any tokens-per-step at a
small micro-batch, which is what `tinystories-small.yaml` does. Computing the
loss in chunks over the sequence (or a fused linear-cross-entropy kernel) removes
the ceiling, but it is an optimisation of something that currently works.

**Trigger.** A CUDA run where the micro-batch is limited by logits rather than
by activations — visible as memory that scales with `vocab_size`, not `n_layers`.
Or a tokenizer with a 128k+ vocabulary, where this gets 2.5x worse.
**Cost later.** Low, and local to the loop's loss line.

### fp16

`training.precision.dtype` accepts `bfloat16` and `float32` only.

**Why deferred.** fp16 has a narrow exponent range, so it needs a loss scaler,
skipped steps on overflow, and care around every reduction. bf16 has fp32's range
and needs none of that, and every GPU this project targets supports it.

**Trigger.** Training on pre-Ampere NVIDIA hardware (V100, T4), which has fp16
but not bf16.
**Cost later.** Moderate: a `GradScaler` in the loop and a second look at each
fp32 upcast.

### Logging backends other than JSONL

The loop writes `log.jsonl` and prints. `utils/logging.py` is still a stub, and
`configs/training/base.yaml` mentions `wandb` and `tensorboard` in a comment.

**Why deferred.** A JSON Lines file is greppable, diffable, plots in three lines,
and has no account or daemon. It is also what an `experiments/` record wants.

**Trigger.** Comparing more runs than is comfortable to overlay by hand.
**Cost later.** Low — the loop calls one `record(entry)` function.

---

## Evaluation

### The whole harness

`eval/harness.py`, `eval/external.py`, `eval/profiling.py`, and all three tasks
are stubs.

**Why deferred.** Explicitly dropped until the swappable-mixer goal is reached.
The four test families — oracle equivalence, causality, state consistency,
single-batch overfit — validate an *implementation* without needing an eval
harness at all. The harness is required for *claims*, which start at the hybrid
ablation.

**Trigger.** The first experiment whose result you want to publish or trust.

---

## Infrastructure

### Registry unification

There are three separate registry implementations: `model/registry.py`,
`data/tokenizer/base.py`, and `data/source.py`. MLPs and norms have none — with
one implementation each, `model/build.py` resolves them from two small dicts.

**Why deferred.** Each registry is ~30 lines and local. Unifying them means
touching model code from data code, or introducing a shared `utils/registry.py`
and migrating all three. Not worth the churn while the interfaces are still
moving.

**Trigger.** A fourth registry, or a change that has to be made in all three.
ROADMAP step 5 adds a second MLP and a second norm; two entries in a dict is
still a dict, but if either grows a third, or wants registration from outside
`build.py`, that is the fourth registry and this fires.
**Cost later.** Low, and mechanical.

### `transformers` is not tested in CI

CI installs `.[dev]`, which includes `tiktoken` but not `transformers`. So
`HFTokenizer` is exercised locally and never in CI — its tests skip on
`ImportError`.

**Why deferred.** `transformers` is a heavy install for a job that would then
also want to download a model over the network.

**Trigger.** A bug in `HFTokenizer` that CI should have caught.
**Cost later.** Low, but it slows every CI run.

### SFT data path

Loss masking, per-example atomicity, padding or block-diagonal packing,
`cu_seqlens`.

The conversation format itself is settled — see *Chat templates* under
Tokenizer. This entry is everything downstream of the rendered string.

**Why deferred.** Genuinely different constraints from pretraining — an example
is atomic, loss covers only the response, and lengths vary wildly. It shares
nothing with `PackedDataset` except the tokenizer. SFT corpora also fit in
memory, so none of the sharding machinery applies.

**Trigger.** Any instruction-tuning work.
**Cost later.** Moderate, but additive — a separate `ExampleDataset`, not a
generalisation of the packed one.
