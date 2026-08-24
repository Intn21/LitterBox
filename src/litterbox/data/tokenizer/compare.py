"""Compare tokenizers on the same text.

The point of a swappable tokenizer interface is being able to ask "which of
these is better for my corpus" and get a number. The number that matters is
**bytes per token**: how much text fits in one model position. Higher is
better — longer effective context, fewer steps to generate the same output.

Everything else here is in service of not being misled by that number. A
tokenizer that scores well while quietly mangling your text is worse than one
that scores badly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from litterbox.data.tokenizer.base import Tokenizer


@dataclass
class TokenizerReport:
    """What one tokenizer did to one piece of text."""

    name: str
    vocab_size: int
    n_tokens: int
    n_bytes: int
    bytes_per_token: float
    lossless: bool
    round_trips: bool
    encode_seconds: float

    @property
    def mb_per_second(self) -> float:
        if self.encode_seconds <= 0:
            return float("inf")
        return self.n_bytes / self.encode_seconds / 1e6


def measure(tok: Tokenizer, text: str) -> TokenizerReport:
    """Run one tokenizer over ``text`` and record what happened."""
    n_bytes = len(text.encode("utf-8"))

    start = time.perf_counter()
    ids = tok.encode(text)
    elapsed = time.perf_counter() - start

    try:
        round_trips = tok.decode(ids) == text
    except UnicodeDecodeError:
        round_trips = False

    return TokenizerReport(
        name=tok.name,
        vocab_size=tok.vocab_size,
        n_tokens=len(ids),
        n_bytes=n_bytes,
        bytes_per_token=n_bytes / len(ids) if ids else 0.0,
        lossless=tok.is_lossless,
        round_trips=round_trips,
        encode_seconds=elapsed,
    )


def compare(tokenizers: Sequence[Tokenizer], text: str) -> list[TokenizerReport]:
    """Measure several tokenizers on the same text, best compression first."""
    reports = [measure(t, text) for t in tokenizers]
    return sorted(reports, key=lambda r: r.bytes_per_token, reverse=True)


def format_comparison(reports: Sequence[TokenizerReport]) -> str:
    """Render reports as a fixed-width table."""
    width = max((len(r.name) for r in reports), default=4)
    head = (
        f"{'tokenizer':<{width}}  {'vocab':>9}  {'tokens':>10}  "
        f"{'bytes/tok':>10}  {'MB/s':>7}  {'round-trip':>10}"
    )
    lines = [head, "-" * len(head)]
    for r in reports:
        lines.append(
            f"{r.name:<{width}}  {r.vocab_size:>9,}  {r.n_tokens:>10,}  "
            f"{r.bytes_per_token:>10.2f}  {r.mb_per_second:>7.1f}  "
            f"{'yes' if r.round_trips else 'NO':>10}"
        )
    return "\n".join(lines)


def print_comparison(tokenizers: Sequence[Tokenizer], text: str) -> list[TokenizerReport]:
    """Compare and print. Returns the reports for further use."""
    reports = compare(tokenizers, text)
    print(format_comparison(reports))
    if any(not r.round_trips for r in reports):
        print("\nNO in round-trip means decode(encode(text)) != text.")
        print("Usually a normalizing tokenizer; never pack a corpus with one.")
    return reports


def show_segmentation(tok: Tokenizer, text: str, *, sep: str = "|") -> str:
    """Show how one tokenizer splits ``text``, piece by piece.

    The qualitative counterpart to bytes-per-token. Two tokenizers can score
    identically and still split text in ways that matter — where the word
    boundaries fall, whether a leading space attaches to the following word,
    how numbers and code get chopped.

    Bytes that don't form a complete character on their own are shown as the
    replacement character, which is itself informative: it marks exactly where
    a multi-byte character was split across tokens.
    """
    pieces = []
    for i in tok.encode(text):
        try:
            pieces.append(tok.decode([i]))
        except UnicodeDecodeError:
            pieces.append("�")
    return sep.join(pieces)


def compare_segmentation(tokenizers: Sequence[Tokenizer], text: str) -> str:
    """Show the same text split by each tokenizer, one per line."""
    width = max((len(t.name) for t in tokenizers), default=4)
    lines = [f"{'':<{width}}  {text!r}", ""]
    for tok in tokenizers:
        n = len(tok.encode(text))
        lines.append(f"{tok.name:<{width}}  {show_segmentation(tok, text)}   ({n} tokens)")
    return "\n".join(lines)
