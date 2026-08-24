"""Reading packed shards during training.

Deliberately not a ``torch.utils.data.Dataset``. For a packed token stream a
batch is a memmap slice and a ``from_numpy`` — DataLoader's worker processes,
collation, and queueing are all overhead on top of what is essentially a memcpy.
nanoGPT skips it for the same reason. You would want it back only for
per-example work such as SFT, which does not use this class anyway.

Three things here are worth being deliberate about, and all three are defences
against failures that produce no error:

- **Construction validates against ``meta.json``.** Hand it a tokenizer and it
  refuses to load shards packed by a different one. Training on a stale ``.bin``
  is otherwise completely invisible.
- **Batches are a function of (seed, step)**, not of how many times you have
  called anything. Resuming from a checkpoint at step 5000 therefore replays the
  same data an uninterrupted run would have seen.
- **Targets are the inputs shifted by one.** Get that wrong and the model learns
  to predict the token it was just shown, which with tied embeddings is nearly
  free — the loss curve looks *better*, not worse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from litterbox.data.pack import PackMeta
from litterbox.data.tokenizer import Tokenizer


class PackedDataset:
    """Random fixed-length windows over a sharded token stream.

    Args:
        path: directory containing ``meta.json`` and the shard files.
        seq_len: tokens per training example. A window needs ``seq_len + 1``
            tokens, since targets are inputs shifted by one.
        split: which split to read.
        tokenizer: optional. If given, its identity is checked against the one
            recorded at pack time and a mismatch raises.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        *,
        split: str = "train",
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.split = split

        self.meta = PackMeta.load(self.path / "meta.json")
        if split not in self.meta.splits:
            raise KeyError(
                f"split {split!r} not in {self.path / 'meta.json'}; "
                f"have: {', '.join(sorted(self.meta.splits))}"
            )
        self.info = self.meta.splits[split]
        self.dtype = np.dtype(self.meta.dtype)
        self.vocab_size = self.meta.vocab_size

        if tokenizer is not None:
            self._check_tokenizer(tokenizer)

        # Shards are memmapped, not read. The OS page cache does the caching,
        # and nothing is copied until a batch actually slices it.
        self._shards = [
            np.memmap(self.path / s.file, dtype=self.dtype, mode="r") for s in self.info.shards
        ]
        for shard, s in zip(self._shards, self.info.shards, strict=True):
            if len(shard) != s.n_tokens:
                raise ValueError(
                    f"{s.file} holds {len(shard):,} tokens but meta.json says "
                    f"{s.n_tokens:,}. The file was truncated, or written with a "
                    f"different dtype than {self.dtype}."
                )

        # A window must fit inside one shard. Sampling across a seam would mean
        # stitching two memmaps for the sake of ~0.001% of start positions.
        need = seq_len + 1
        self._valid = np.array([max(0, s.n_tokens - need + 1) for s in self.info.shards])
        self._cum = np.cumsum(self._valid)
        self.n_valid_positions = int(self._cum[-1]) if len(self._cum) else 0

        if self.n_valid_positions == 0:
            raise ValueError(
                f"no shard in split {split!r} holds {need:,} tokens, so no window "
                f"of seq_len={seq_len} can be sampled"
            )

    # ------------------------------------------------------------ validation

    def _check_tokenizer(self, tokenizer: Tokenizer) -> None:
        if tokenizer.name != self.meta.tokenizer_name:
            raise ValueError(
                f"{self.path} was packed with {self.meta.tokenizer_name!r} but you "
                f"passed {tokenizer.name!r}. The ids in those shards mean something "
                f"different to this tokenizer; nothing downstream would notice."
            )
        if tokenizer.vocab_size != self.meta.vocab_size:
            raise ValueError(
                f"vocab_size mismatch: shards say {self.meta.vocab_size}, "
                f"tokenizer says {tokenizer.vocab_size}"
            )

    # --------------------------------------------------------------- sizes

    def __len__(self) -> int:
        """Total tokens in this split."""
        return self.info.n_tokens

    @property
    def n_documents(self) -> int:
        return self.info.n_documents

    def __repr__(self) -> str:
        return (
            f"<PackedDataset {self.split!r} {self.info.n_tokens:,} tokens "
            f"in {len(self._shards)} shard(s), {self.meta.dtype}, "
            f"tokenizer={self.meta.tokenizer_name!r}>"
        )

    # -------------------------------------------------------------- batching

    def _locate(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map global valid-position indices to (shard index, offset in shard)."""
        shard_idx = np.searchsorted(self._cum, flat, side="right")
        base = self._cum[shard_idx] - self._valid[shard_idx]
        return shard_idx, flat - base

    def get_batch(
        self,
        batch_size: int,
        step: int,
        *,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one batch of windows.

        Args:
            batch_size: rows in the batch.
            step: training step. Together with ``seed`` this fully determines
                the batch, so a resumed run replays exactly.
            seed: run seed.
            device: where to put the tensors. CUDA transfers are pinned and
                non-blocking.

        Returns:
            ``(x, y)``, each ``[batch_size, seq_len]`` of int64, with ``y``
            being ``x`` shifted one position forward.
        """
        # Derived from (seed, step) rather than advancing a stateful generator:
        # call count must not affect which data a given step sees.
        rng = np.random.default_rng([seed, step])
        flat = rng.integers(0, self.n_valid_positions, size=batch_size)
        shard_idx, offset = self._locate(flat)

        need = self.seq_len + 1
        window = np.empty((batch_size, need), dtype=np.int64)
        for row, (si, off) in enumerate(zip(shard_idx, offset, strict=True)):
            window[row] = self._shards[si][off : off + need]

        x = torch.from_numpy(window[:, :-1])
        y = torch.from_numpy(window[:, 1:])

        if device is not None:
            device = torch.device(device)
            if device.type == "cuda":
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
        return x, y

    # ---------------------------------------------------------------- oracle

    def read(self, start: int, length: int) -> np.ndarray:
        """Read ``length`` tokens from global position ``start``, across shards.

        For inspection and tests rather than training — this is the method that
        lets you decode a slice and check that packing did not scramble
        anything.
        """
        out = np.empty(length, dtype=np.int64)
        written = 0
        pos = start
        for shard in self._shards:
            if pos >= len(shard):
                pos -= len(shard)
                continue
            take = min(length - written, len(shard) - pos)
            out[written : written + take] = shard[pos : pos + take]
            written += take
            pos = 0
            if written == length:
                return out
        raise ValueError(f"requested {length} tokens from {start}, only {written} available")

    def document_offsets(self) -> np.ndarray | None:
        """Global token index where each document starts, if it was written."""
        path = self.path / f"{self.split}_docs.bin"
        if not path.exists():
            return None
        return np.fromfile(path, dtype=np.int64)
