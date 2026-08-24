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

---

## Model

### Everything

25 files under `src/litterbox/` still raise `NotImplementedError`: all seven
mixers, the backbone, the block, RoPE, generation, the cache, and training.

An implementation was written and then **deliberately reverted** (`64e5658`,
reverted by `a50c74b`) — building it is the practice this project exists for.
The interface, the registry, and the configs stand; the bodies do not.

**Trigger.** Now. Steps 1 through 6 in [ROADMAP.md](ROADMAP.md), starting with
`utils/config.py`.

### `positions` tensor instead of `pos_offset: int`

`TokenMixer.forward` takes `pos_offset: int`, which assumes positions are
contiguous from an offset. True for every case in this project today.

**What breaks it.** Packed SFT, where several examples share one sequence and
each must restart at position 0. An int cannot express that.

**Trigger.** An SFT path, or packed variable-length training.
**Cost later.** Mechanical but wide — the signature change touches every mixer.

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
`data/tokenizer/base.py`, and `data/source.py`.

**Why deferred.** Each is ~30 lines and local. Unifying them means touching model
code from data code, or introducing a shared `utils/registry.py` and migrating
all three. Not worth the churn while the interfaces are still moving.

**Trigger.** A fourth registry, or a change that has to be made in all three.
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

**Why deferred.** Genuinely different constraints from pretraining — an example
is atomic, loss covers only the response, and lengths vary wildly. It shares
nothing with `PackedDataset` except the tokenizer. SFT corpora also fit in
memory, so none of the sharding machinery applies.

**Trigger.** Any instruction-tuning work.
**Cost later.** Moderate, but additive — a separate `ExampleDataset`, not a
generalisation of the packed one.
