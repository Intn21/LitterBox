"""Turn a text source into sharded token files plus a metadata sidecar.

This is the offline job, run once per (corpus, tokenizer) pair. It is also the
only part of the data path where efficiency is worth thinking about: reading a
batch at training time is a page-cached memmap slice measured in microseconds,
while tokenizing a corpus is CPU-bound and can take hours.

So the one optimisation that matters lives here — a process pool over documents.
Workers build their own tokenizer from a config rather than receiving a pickled
one, which avoids both per-task serialisation and backends that do not pickle.

Two silent-failure guards run at pack time, because neither raises on its own:

- the dtype is derived from ``vocab_size``, never assumed. A vocabulary above
  65,535 stored as ``uint16`` wraps modulo 65,536 and turns real tokens into
  different, equally valid-looking ones.
- ``meta.json`` records which tokenizer produced the file. A bare array of
  integers cannot tell you how to read it, and training on a stale shard
  produces no error at all — just a worse loss curve you blame on the model.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np

from litterbox.data.tokenizer import Tokenizer, build_tokenizer

META_VERSION = 1
DEFAULT_SHARD_TOKENS = 100_000_000  # ~200MB at uint16, ~400MB at uint32


# --------------------------------------------------------------------- meta


@dataclass
class ShardInfo:
    file: str
    n_tokens: int


@dataclass
class SplitInfo:
    shards: list[ShardInfo] = field(default_factory=list)
    n_tokens: int = 0
    n_documents: int = 0


@dataclass
class PackMeta:
    """Everything needed to read the shards back, and to prove they are yours."""

    version: int
    tokenizer_name: str
    vocab_size: int
    dtype: str
    eos_id: int | None
    eos_between_docs: bool
    tokenizer: dict | None = None  # config, so the tokenizer can be rebuilt
    splits: dict[str, SplitInfo] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def load(cls, path: Path) -> PackMeta:
        raw = json.loads(Path(path).read_text())
        splits = {
            name: SplitInfo(
                shards=[ShardInfo(**s) for s in info["shards"]],
                n_tokens=info["n_tokens"],
                n_documents=info["n_documents"],
            )
            for name, info in raw.pop("splits", {}).items()
        }
        return cls(splits=splits, **raw)


def choose_dtype(vocab_size: int) -> np.dtype:
    """The smallest unsigned type that can hold every id in the vocabulary.

    Not a size optimisation you opt into — a correctness constraint. ``uint16``
    tops out at 65,535, so GPT-2 (50,257) fits while Llama-3 (128,256) and
    o200k (200,019) do not.
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size <= 2**16:
        return np.dtype(np.uint16)
    if vocab_size <= 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"vocab_size {vocab_size:,} exceeds uint32")


# ------------------------------------------------------------------ workers

_WORKER_TOKENIZER: Tokenizer | None = None


def _init_worker(cfg: dict) -> None:
    """Build one tokenizer per worker process, once."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = build_tokenizer(cfg)


def _encode_in_worker(texts: list[str]) -> list[list[int]]:
    """Encode a whole group per task.

    Per-document tasks spend more time pickling strings than tokenizing them —
    tiktoken encodes at roughly 50 MB/s, so IPC is the bottleneck, not the work.
    Sending groups amortises that.
    """
    assert _WORKER_TOKENIZER is not None, "worker was not initialised"
    return _WORKER_TOKENIZER.encode_batch(texts)


def _chunked(iterable: Iterable[str], n: int) -> Iterator[list[str]]:
    """Group a lazy iterable into lists of ``n``, staying lazy."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def _pool_is_safe() -> str:
    """Return a reason a process pool cannot start here, or "" if it can.

    macOS and Windows default to the "spawn" start method, where each worker
    re-imports ``__main__``. From a REPL, a notebook, or a heredoc there is no
    importable ``__main__``, so every worker dies on import and is replaced —
    an unbounded spawn loop that survives the parent.

    Python already raises a clear error for the related case of a script
    without an ``if __name__ == "__main__":`` guard. It does not catch this
    one, so we do.
    """
    if mp.get_start_method(allow_none=True) == "fork":
        return ""
    main = sys.modules.get("__main__")
    if getattr(main, "__file__", None) is None:
        return (
            "this interpreter has no importable __main__ (REPL, notebook, or "
            "piped stdin) and the platform uses the 'spawn' start method"
        )
    return ""


# ------------------------------------------------------------ shard writing


class _ShardWriter:
    """Accumulates ids in a fixed buffer, flushing a file each time it fills.

    The buffer is what keeps memory flat regardless of corpus size: one shard's
    worth at a time, never the whole stream.
    """

    def __init__(self, out_dir: Path, split: str, dtype: np.dtype, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.dtype = dtype
        self.buf = np.empty(shard_tokens, dtype=dtype)
        self.n = 0
        self.shards: list[ShardInfo] = []

    def add(self, ids: np.ndarray) -> None:
        """Append ids, flushing whenever the buffer fills.

        Handles documents longer than a whole shard by spilling across as many
        as needed, so no corpus-dependent assumption creeps in.
        """
        pos = 0
        while pos < len(ids):
            room = len(self.buf) - self.n
            take = min(room, len(ids) - pos)
            self.buf[self.n : self.n + take] = ids[pos : pos + take]
            self.n += take
            pos += take
            if self.n == len(self.buf):
                self.flush()

    def flush(self) -> None:
        if self.n == 0:
            return
        name = f"{self.split}_{len(self.shards):04d}.bin"
        self.buf[: self.n].tofile(self.out_dir / name)
        self.shards.append(ShardInfo(file=name, n_tokens=int(self.n)))
        self.n = 0

    @property
    def total_tokens(self) -> int:
        return sum(s.n_tokens for s in self.shards)


# --------------------------------------------------------------------- pack


def _encoded_documents(
    source: Iterable[str],
    tokenizer: Tokenizer,
    tokenizer_cfg: dict | None,
    n_workers: int,
    chunk_size: int,
) -> Iterator[list[int]]:
    """Yield encoded documents, in parallel where that is possible.

    A pool needs a config to rebuild the tokenizer inside each worker. Without
    one — a tokenizer trained in memory, say — we fall back to serial rather
    than trying to pickle it.
    """
    if n_workers > 1 and tokenizer_cfg is None:
        warnings.warn(
            "n_workers > 1 needs tokenizer_cfg so each worker can rebuild the "
            "tokenizer; falling back to serial encoding.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif n_workers > 1 and (reason := _pool_is_safe()):
        warnings.warn(
            f"falling back to serial encoding: {reason}. Call pack() from a "
            f'script guarded by `if __name__ == "__main__":` to use workers.',
            RuntimeWarning,
            stacklevel=3,
        )
        n_workers = 1

    groups = _chunked(source, chunk_size)
    if n_workers > 1 and tokenizer_cfg is not None:
        with mp.Pool(n_workers, initializer=_init_worker, initargs=(tokenizer_cfg,)) as pool:
            for encoded in pool.imap(_encode_in_worker, groups):
                yield from encoded
    else:
        # Not a fallback so much as the fast default: encode_batch lets tiktoken
        # and HF fan out across Rust threads with no process or pickling cost.
        for group in groups:
            yield from tokenizer.encode_batch(group)


def pack(
    source: Iterable[str],
    tokenizer: Tokenizer,
    out_dir: str | Path,
    *,
    split: str = "train",
    tokenizer_cfg: dict | None = None,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    n_workers: int | None = None,
    eos_between_docs: bool = True,
    write_doc_offsets: bool = True,
    chunk_size: int = 256,
    progress_every: int | None = None,
) -> PackMeta:
    """Tokenize ``source`` into ``out_dir`` as sharded ``.bin`` files.

    Args:
        source: an iterable of documents. Kept lazy — never materialised.
        tokenizer: used for ``vocab_size``, ``eos_id``, and serial encoding.
        out_dir: written to. Created if absent.
        split: name for this pass. Call twice with ``"train"`` and ``"val"`` on
            disjoint documents; splitting one packed stream instead lets a
            validation window reach into training data.
        tokenizer_cfg: config that rebuilds ``tokenizer``. Required for the
            process pool, and recorded in ``meta.json``.
        shard_tokens: tokens per shard. Bounds memory during packing and the
            damage a crash can do.
        n_workers: processes for tokenization. Defaults to 1, which is *not*
            the slow path — serial encoding goes through ``encode_batch``, so
            tiktoken and HF still fan out across Rust threads without any
            process or pickling cost.

            Measured on a 10-core M-series with GPT-2: batched serial reaches
            ~75 MB/s of text. A 4-worker pool on 192 MB managed ~80 MB/s, or
            1.08x — barely worth the spawn hazard. Against *unbatched* serial
            the same pool looked like 1.8x, which is how process pools get
            cargo-culted into pipelines that did not need them.

            So leave this at 1 unless you have measured otherwise on your own
            corpus and hardware. If you do raise it, 4 beat 8 in every run.

        eos_between_docs: insert ``eos_id`` between documents so the model has a
            boundary signal. Off only if your tokenizer has no EOS.
        write_doc_offsets: also write ``{split}_docs.npy``, the global token
            index where each document starts. Costs ~8 bytes per document and
            is impossible to recover later without repacking, which is why it
            defaults on even though nothing reads it yet.

    Returns:
        The :class:`PackMeta` that was written to ``meta.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = choose_dtype(tokenizer.vocab_size)
    if n_workers is None:
        n_workers = 1  # see the docstring: a pool loses below ~50MB of text

    eos_id = tokenizer.eos_id
    if eos_between_docs and eos_id is None:
        eos_between_docs = False

    writer = _ShardWriter(out_dir, split, dtype, shard_tokens)
    doc_starts: list[int] = []
    n_docs = 0
    running = 0

    for ids in _encoded_documents(source, tokenizer, tokenizer_cfg, n_workers, chunk_size):
        if eos_between_docs:
            ids = [*ids, eos_id]
        if not ids:
            continue

        arr = np.asarray(ids, dtype=np.int64)
        # The wraparound guard. Casting a too-large id is silent and produces a
        # different, perfectly valid token — see docs on choose_dtype.
        biggest = int(arr.max())
        if biggest >= tokenizer.vocab_size:
            raise ValueError(
                f"token id {biggest} >= vocab_size {tokenizer.vocab_size}; "
                f"storing it as {dtype} would wrap silently"
            )

        doc_starts.append(running)
        running += len(arr)
        n_docs += 1
        writer.add(arr.astype(dtype))

        if progress_every and n_docs % progress_every == 0:
            print(f"  {n_docs:,} docs, {running:,} tokens", flush=True)

    writer.flush()

    if write_doc_offsets and doc_starts:
        np.asarray(doc_starts, dtype=np.int64).tofile(out_dir / f"{split}_docs.bin")

    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = PackMeta.load(meta_path)
        if meta.tokenizer_name != tokenizer.name:
            raise ValueError(
                f"{meta_path} was packed with {meta.tokenizer_name!r}, "
                f"but this call uses {tokenizer.name!r}. Mixing tokenizers in one "
                f"dataset directory would be undetectable at read time."
            )
    else:
        meta = PackMeta(
            version=META_VERSION,
            tokenizer_name=tokenizer.name,
            vocab_size=tokenizer.vocab_size,
            dtype=dtype.name,
            eos_id=eos_id,
            eos_between_docs=eos_between_docs,
            tokenizer=tokenizer_cfg,
        )

    meta.splits[split] = SplitInfo(
        shards=writer.shards,
        n_tokens=writer.total_tokens,
        n_documents=n_docs,
    )
    meta_path.write_text(meta.to_json())
    return meta
