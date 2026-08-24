"""Where raw text comes from.

A source is just an ``Iterable[str]`` — one string per document. That is a low
enough bar that anything can be one, and it keeps the packer from caring whether
text arrived from the Hub, from disk, or from a test fixture.

Sources are lazy on purpose. A pretraining corpus does not fit in memory, and
``pack()`` only ever holds one shard's worth at a time; a source that
materialised its documents into a list would undo that.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

SourceFn = Callable[..., Iterator[str]]
_SOURCES: dict[str, SourceFn] = {}


def register_source(name: str) -> Callable[[SourceFn], SourceFn]:
    """Register a source builder under ``name`` so configs can name it."""

    def decorator(fn: SourceFn) -> SourceFn:
        if name in _SOURCES and _SOURCES[name] is not fn:
            raise ValueError(f"source {name!r} is already registered")
        _SOURCES[name] = fn
        return fn

    return decorator


def available_sources() -> list[str]:
    return sorted(_SOURCES)


def build_source(cfg: dict) -> Iterator[str]:
    """Construct a source from a config block: ``{"type": "files", ...}``."""
    cfg = dict(cfg)
    name = cfg.pop("type")
    try:
        builder = _SOURCES[name]
    except KeyError:
        raise KeyError(
            f"unknown source {name!r}; available: {', '.join(available_sources())}"
        ) from None
    return builder(**cfg)


@register_source("text")
def from_text(text: str | list[str]) -> Iterator[str]:
    """One or more documents held in memory. For tests and small experiments."""
    yield from ([text] if isinstance(text, str) else text)


@register_source("files")
def from_files(
    pattern: str,
    *,
    root: str | Path = ".",
    encoding: str = "utf-8",
    per_line: bool = False,
) -> Iterator[str]:
    """Documents from local files matching a glob.

    Args:
        pattern: glob relative to ``root``, e.g. ``"corpus/*.txt"``.
        per_line: treat each line as its own document rather than each file.
            Set this for ``.jsonl``-adjacent line-delimited plain text.
    """
    for path in sorted(Path(root).glob(pattern)):
        if per_line:
            with path.open(encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        else:
            yield path.read_text(encoding=encoding)


@register_source("jsonl")
def from_jsonl(
    pattern: str,
    *,
    root: str | Path = ".",
    text_key: str = "text",
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Documents from JSON Lines files, reading one field per record."""
    for path in sorted(Path(root).glob(pattern)):
        with path.open(encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)[text_key]


@register_source("huggingface")
def from_huggingface(
    name: str,
    *,
    split: str = "train",
    text_key: str = "text",
    streaming: bool = True,
    limit: int | None = None,
    **kwargs,
) -> Iterator[str]:
    """Documents from a HuggingFace dataset.

    Streams by default so a large corpus never lands on disk twice or in memory
    at all. ``limit`` caps the document count, which is how you take a slice of
    something enormous for a smoke test.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("the 'huggingface' source needs: pip install datasets") from e

    ds = load_dataset(name, split=split, streaming=streaming, **kwargs)
    for n, record in enumerate(ds):
        if limit is not None and n >= limit:
            return
        yield record[text_key]
