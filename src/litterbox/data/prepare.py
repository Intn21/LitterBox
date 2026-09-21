"""From a data config to a packed dataset directory, in one command.

    python -m litterbox.data.prepare configs/data/tinystories.yaml
    litterbox-pack configs/data/tinystories.yaml --limit 2000      # smoke test

``pack()`` turns one stream of documents into one split. This module is the
layer above it: a YAML file names the corpus, the tokenizer, and the splits, and
``prepare()`` produces the whole directory a training run reads.

A data config comes in two shapes. When the corpus ships its own splits, name
each one::

    tokenizer: {type: tiktoken, encoding: gpt2}
    out_dir: data/tinystories-gpt2
    splits:
      train: {type: huggingface, name: roneneldan/TinyStories, split: train}
      val:   {type: huggingface, name: roneneldan/TinyStories, split: validation}

When it does not, give one source and carve a validation set out of it by
document hash (see :func:`litterbox.data.source.holdout`)::

    source: {type: huggingface, name: HuggingFaceFW/fineweb-edu, subset: sample-10BT}
    holdout: {val_fraction: 0.005}

**Packing is expensive, so it must be safe to re-run.** Tokenizing a corpus takes
minutes to hours, and a training script will call this on every launch. So the
directory carries a ``dataset.json`` sidecar holding a fingerprint of the config
that produced it, and ``prepare()`` does one of exactly three things:

- the directory is absent: pack it;
- it is present with a matching fingerprint: do nothing;
- it is present and differs, or has no sidecar: **refuse**. Tokens from one
  config silently standing in for another is the stale-shard failure again, one
  level up. ``force=True`` repacks — and deletes only a directory that is
  recognisably a packed dataset, never an arbitrary path from a typo.

The sidecar is written last, so a run that died halfway leaves a directory with
no sidecar, which reads as "not mine" rather than "up to date".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litterbox.data.pack import DEFAULT_SHARD_TOKENS, PackMeta, pack
from litterbox.data.source import build_source, holdout
from litterbox.data.tokenizer import build_tokenizer

SIDECAR = "dataset.json"
SIDECAR_VERSION = 1


# ------------------------------------------------------------------- schema


class _Strict(BaseModel):
    # A misspelled key must fail at load time, not quietly fall back to a
    # default and produce a dataset that is subtly not the one you asked for.
    model_config = ConfigDict(extra="forbid")


class HoldoutConfig(_Strict):
    val_fraction: float = Field(gt=0.0, lt=1.0)
    seed: int = 0


class DataConfig(_Strict):
    """One corpus, one tokenizer, one output directory.

    ``source`` / ``splits`` / ``tokenizer`` blocks are passed through to
    :func:`build_source` and :func:`build_tokenizer`, so whatever those accept
    is valid here and a new registered source needs no schema change.
    """

    out_dir: str
    tokenizer: dict[str, Any]
    splits: dict[str, dict[str, Any]] | None = None
    source: dict[str, Any] | None = None
    holdout: HoldoutConfig | None = None
    limit: int | None = Field(default=None, gt=0)
    shard_tokens: int = Field(default=DEFAULT_SHARD_TOKENS, gt=0)
    description: str | None = None

    @model_validator(mode="after")
    def _one_shape(self) -> DataConfig:
        if (self.splits is None) == (self.source is None):
            raise ValueError(
                "give exactly one of `splits` (the corpus ships its own) or "
                "`source` + `holdout` (carve a validation set out of one stream)"
            )
        if self.source is not None and self.holdout is None:
            raise ValueError(
                "`source` needs a `holdout` block: a run with no validation split cannot "
                "tell learning from memorising. Use `splits: {train: ...}` to opt out explicitly"
            )
        if self.splits is not None and self.holdout is not None:
            raise ValueError("`holdout` only applies to `source`; `splits` are already separate")
        if self.splits is not None and "train" not in self.splits:
            raise ValueError(f"`splits` must include 'train', got {sorted(self.splits)}")
        return self

    def fingerprint(self) -> str:
        """Hash of everything that changes the bytes on disk.

        ``out_dir`` and ``description`` are excluded: moving a dataset or
        rewording its comment does not make it a different dataset.
        """
        payload = self.model_dump(exclude={"out_dir", "description"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_data_config(path: str | Path, **overrides: Any) -> DataConfig:
    """Read and strictly validate a data config. ``overrides`` replace top-level keys."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return DataConfig.model_validate(raw)


# ------------------------------------------------------------------ sources


def _split_sources(cfg: DataConfig) -> dict[str, Iterator[str]]:
    """One lazy document stream per split. Nothing is read until ``pack()`` pulls."""
    if cfg.splits is not None:
        streams = {name: build_source(block) for name, block in cfg.splits.items()}
    else:
        assert cfg.source is not None and cfg.holdout is not None
        h = cfg.holdout
        streams = {
            # Built twice on purpose: each side is its own independent pass.
            keep: holdout(
                build_source(cfg.source), val_fraction=h.val_fraction, keep=keep, seed=h.seed
            )
            for keep in ("train", "val")
        }
    if cfg.limit is not None:
        streams = {name: islice(stream, cfg.limit) for name, stream in streams.items()}
    return streams


# ------------------------------------------------------------------ prepare


def _looks_like_a_dataset(path: Path) -> bool:
    return (path / SIDECAR).exists() or (path / "meta.json").exists()


def is_prepared(cfg: DataConfig) -> bool:
    """True when ``cfg.out_dir`` already holds exactly this config's output."""
    out = Path(cfg.out_dir)
    try:
        sidecar = json.loads((out / SIDECAR).read_text())
        meta = PackMeta.load(out / "meta.json")
    except FileNotFoundError:
        return False
    expected = set(cfg.splits) if cfg.splits is not None else {"train", "val"}
    return sidecar.get("fingerprint") == cfg.fingerprint() and expected <= set(meta.splits)


def prepare(cfg: DataConfig, *, force: bool = False, progress: bool = True) -> PackMeta:
    """Make ``cfg.out_dir`` hold the packed dataset ``cfg`` describes.

    Idempotent: a directory that already matches is returned untouched. See the
    module docstring for the three cases.

    Raises:
        FileExistsError: ``out_dir`` holds something else and ``force`` is off,
            or ``force`` is on but the directory is not a packed dataset.
    """
    out = Path(cfg.out_dir)
    say = print if progress else (lambda *a, **k: None)

    if is_prepared(cfg):
        say(f"{out}: up to date (fingerprint {cfg.fingerprint()}), nothing to do")
        return PackMeta.load(out / "meta.json")

    if out.exists() and any(out.iterdir()):
        if not force:
            why = (
                "was packed from a different config"
                if (out / SIDECAR).exists()
                else f"has no {SIDECAR} (packed by hand, or a previous run died partway)"
            )
            raise FileExistsError(
                f"{out} already exists and {why}. Refusing to mix datasets in one directory. "
                f"Re-run with force=True / --force to delete it and repack, or change out_dir."
            )
        if not _looks_like_a_dataset(out):
            raise FileExistsError(
                f"{out} is not empty and does not look like a packed dataset (no meta.json or "
                f"{SIDECAR}), so --force will not delete it. Check out_dir for a typo."
            )
        say(f"{out}: removing the existing dataset (--force)")
        shutil.rmtree(out)

    tokenizer = build_tokenizer(cfg.tokenizer)
    say(f"{out}: packing with {tokenizer.name} (vocab {tokenizer.vocab_size:,})")
    started = time.perf_counter()
    meta = None
    for split, stream in _split_sources(cfg).items():
        t0 = time.perf_counter()
        meta = pack(
            stream,
            tokenizer,
            out,
            split=split,
            tokenizer_cfg=cfg.tokenizer,
            shard_tokens=cfg.shard_tokens,
            progress_every=200_000 if progress else None,
        )
        info = meta.splits[split]
        say(
            f"  {split:>5}: {info.n_documents:>10,} docs  {info.n_tokens:>14,} tokens  "
            f"{len(info.shards)} shard(s)  {time.perf_counter() - t0:,.1f}s"
        )
    assert meta is not None

    # Last, so that its presence means "every split finished".
    (out / SIDECAR).write_text(
        json.dumps(
            {
                "version": SIDECAR_VERSION,
                "fingerprint": cfg.fingerprint(),
                "config": cfg.model_dump(),
                "seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        )
        + "\n"
    )
    return meta


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="litterbox-pack",
        description="Tokenize a corpus into packed training shards, from a data config.",
    )
    parser.add_argument("config", help="path to a data config, e.g. configs/data/tinystories.yaml")
    parser.add_argument(
        "--force", action="store_true", help="delete and repack a directory that does not match"
    )
    parser.add_argument(
        "--limit", type=int, help="cap documents per split (smoke test); changes the fingerprint"
    )
    parser.add_argument("--out", dest="out_dir", help="override the config's out_dir")
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config, limit=args.limit, out_dir=args.out_dir)
    try:
        meta = prepare(cfg, force=args.force)
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    total = sum(s.n_tokens for s in meta.splits.values())
    size = sum(f.stat().st_size for f in Path(cfg.out_dir).glob("*.bin"))
    print(f"{cfg.out_dir}: {total:,} tokens as {meta.dtype}, {size / 1e6:,.1f} MB on disk")
    return 0


if __name__ == "__main__":  # the guard matters: pack()'s workers re-import this module
    raise SystemExit(main())
