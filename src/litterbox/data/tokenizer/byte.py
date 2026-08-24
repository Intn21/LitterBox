"""Byte and character tokenizers — no training, no dependencies, no network.

These are the baselines. A byte tokenizer compresses nothing (one token per
byte, by definition), which is exactly what makes it useful: every other
tokenizer's bytes-per-token is measured against this floor.

It is also the only tokenizer guaranteed to work offline with nothing
installed, so tests and examples can depend on it unconditionally.
"""

from __future__ import annotations

from collections.abc import Sequence

from litterbox.data.tokenizer.base import Tokenizer, register_tokenizer


@register_tokenizer("byte")
class ByteTokenizer(Tokenizer):
    """One token per UTF-8 byte. Vocabulary is exactly 256.

    Lossless for any input, and incapable of producing an unknown token, since
    every possible byte already has an id. That property is what byte-level BPE
    inherits by starting from this vocabulary.
    """

    def __init__(self) -> None:
        self.name = "byte"
        self.vocab_size = 256
        self.bos_id = None
        self.eos_id = None
        self._validate()

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        # Join first, decode once. Decoding per id would fail on any multi-byte
        # character, since a continuation byte means nothing on its own.
        return bytes(ids).decode("utf-8", errors="replace")


@register_tokenizer("char")
class CharTokenizer(Tokenizer):
    """One token per Unicode character, over a fixed alphabet.

    Included for contrast with ``byte``: it needs an alphabet, and anything
    outside that alphabet is unrepresentable. That failure mode is the whole
    argument for working at the byte level instead.

    Args:
        alphabet: characters to cover. Defaults to printable ASCII.
        unknown: character substituted for anything outside the alphabet.
    """

    def __init__(self, alphabet: str | None = None, unknown: str = "�") -> None:
        if alphabet is None:
            alphabet = "".join(chr(i) for i in range(32, 127)) + "\n\t"

        self.itos = [unknown, *sorted(set(alphabet))]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.unknown_id = 0

        self.name = f"char[{len(self.itos)}]"
        self.vocab_size = len(self.itos)
        self.bos_id = None
        self.eos_id = None
        self._validate()

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unknown_id) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids)
