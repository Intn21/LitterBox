"""Text to training batches, and what it costs.

Run it::

    python examples/data_pipeline.py            # demo + a small benchmark
    python examples/data_pipeline.py --big      # 200MB, closer to real numbers

The numbers printed at the end are hardware-specific. Re-run this on whatever
you actually train on rather than trusting figures measured elsewhere — the
serial/parallel tradeoff in particular flips depending on core count and
multiprocessing start method.

NOTE the ``if __name__ == "__main__"`` guard at the bottom. It is not optional
when packing with workers: on macOS and Windows each worker re-imports this
module, and without the guard that re-runs the packing call.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from litterbox.data import PackedDataset, pack
from litterbox.data.tokenizer import build_tokenizer

TOKENIZER = {"type": "tiktoken", "encoding": "gpt2"}

PARAGRAPH = (
    "The researchers examined how tokenization affects downstream model quality. "
    "They found that vocabulary size interacts with corpus domain in ways that are "
    "not obvious from compression ratio alone. A larger vocabulary compresses better "
    "on in-domain text but spends embedding capacity on tokens that rarely appear. "
)


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * len(title))


def corpus(n_docs: int) -> list[str]:
    return [PARAGRAPH * 6 + f" This is document number {i}." for i in range(n_docs)]


def main(big: bool = False) -> None:
    tok = build_tokenizer(TOKENIZER)
    out = Path(tempfile.mkdtemp(prefix="litterbox-data-"))
    n_docs = 80_000 if big else 8_000

    try:
        # ── pack ──────────────────────────────────────────────────────────
        section("1. Packing text into shards")
        docs = corpus(n_docs)
        mb = sum(len(d.encode()) for d in docs) / 1e6
        print(f"  {len(docs):,} documents, {mb:.1f} MB of text")
        print(f"  tokenizer: {tok.name}  (vocab {tok.vocab_size:,})")

        t0 = time.perf_counter()
        meta = pack(docs, tok, out, tokenizer_cfg=TOKENIZER, shard_tokens=5_000_000)
        elapsed = time.perf_counter() - t0

        info = meta.splits["train"]
        disk = sum(f.stat().st_size for f in out.glob("*.bin")) / 1e6
        print(f"\n  packed in {elapsed:.2f}s  ({mb / elapsed:.0f} MB/s)")
        print(f"  {info.n_tokens:,} tokens across {len(info.shards)} shards -> {disk:.1f} MB")
        print(f"  dtype {meta.dtype}, chosen from vocab_size (not assumed)")
        print(f"  compression: {mb * 1e6 / info.n_tokens:.2f} bytes/token")

        # ── metadata ──────────────────────────────────────────────────────
        section("2. The sidecar is what makes shards readable")
        print(f"  {out / 'meta.json'} records:")
        for k in ("tokenizer_name", "vocab_size", "dtype", "eos_id"):
            print(f"    {k:<16} {getattr(meta, k)}")
        print("\n  A .bin is raw bytes — it cannot say which tokenizer wrote it,")
        print("  or whether to read it two bytes at a time or four.")

        # ── batching ──────────────────────────────────────────────────────
        section("3. Reading batches")
        ds = PackedDataset(out, seq_len=1024, tokenizer=tok)
        print(f"  {ds}")
        print(f"  {ds.n_valid_positions:,} valid window starts")

        x, y = ds.get_batch(8, step=0)
        print(f"\n  x {tuple(x.shape)}   y {tuple(y.shape)}   {x.dtype}")
        print(f"  y is x shifted by one: {bool((x[:, 1:] == y[:, :-1]).all())}")
        print("\n  If that were False the model would learn to predict the token")
        print("  it was just shown — and the loss curve would look better, not worse.")

        # ── determinism ───────────────────────────────────────────────────
        section("4. Batches are a function of (seed, step)")
        a, _ = ds.get_batch(4, step=42)
        b, _ = ds.get_batch(4, step=42)
        c, _ = ds.get_batch(4, step=43)
        print(f"  same step, twice   -> identical: {bool((a == b).all())}")
        print(f"  next step          -> different: {not bool((a == c).all())}")
        print("\n  So resuming from a checkpoint at step N replays exactly what an")
        print("  uninterrupted run would have seen. No stateful generator to desync.")

        # ── oracle ────────────────────────────────────────────────────────
        section("5. Oracle: decode the stream back")
        print(f"  {tok.decode(ds.read(0, 24).tolist())!r}")
        offsets = ds.document_offsets()
        if offsets is not None:
            print(f"\n  {len(offsets):,} document offsets recorded "
                  f"(first few: {offsets[:4].tolist()})")
            print("  Unused for now — but needed for intra-document attention")
            print("  masking, and impossible to recover without repacking.")

        # ── performance ───────────────────────────────────────────────────
        section("6. What reading actually costs")
        print(f"  {'batch':<18} {'us/batch':>10} {'batches/s':>11} {'Mtok/s':>9}")
        print("  " + "-" * 50)
        for seq_len, bs in ((512, 16), (1024, 32), (2048, 32)):
            d = PackedDataset(out, seq_len=seq_len)
            for s in range(20):
                d.get_batch(bs, step=s)
            t0 = time.perf_counter()
            n = 200
            for s in range(n):
                d.get_batch(bs, step=10_000 + s)
            dt = (time.perf_counter() - t0) / n
            label = f"B={bs} S={seq_len}"
            mtok = bs * seq_len / dt / 1e6
            print(f"  {label:<18} {dt * 1e6:>10.0f} {1 / dt:>11.0f} {mtok:>9.1f}")

        print("\n  A forward+backward pass on B=32 S=1024 for a ~100M model is tens")
        print("  of milliseconds. Loading is microseconds — well under 1% of a step.")
        print("  That is why this is a memmap slice and not a DataLoader.")

    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main(big="--big" in sys.argv)
