"""The tokenizer interface, and the contract every backend is forced into.

Three libraries wrap tokenizers and all three disagree, silently, about things
that matter: whether ``encode`` adds a BOS token, what ``vocab_size`` counts,
and whether decoding round-trips. A wrapper that merely forwards calls inherits
those disagreements and surfaces them somewhere expensive — halfway through
packing a corpus, or as an index error deep in training.

So this module does two things instead. It picks one canonical behaviour, and it
*verifies at construction* that the backend actually complied. The check costs a
few milliseconds once per tokenizer.

The canonical rules:

1. ``encode`` returns **content tokens only**. Never BOS, never EOS. Special
   tokens are exposed as data (``bos_id``, ``eos_id``) and added explicitly by
   the caller, so a swap between tokenizers can never silently shift a dataset
   by one position.
2. ``vocab_size`` is **the number the embedding matrix must have** — the count
   including added tokens, not the base vocabulary.
3. ``is_lossless`` is **measured, not declared**. Byte-level BPE round-trips;
   normalizing tokenizers do not. Which one you have is determined empirically
   at construction rather than hardcoded per backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

# Deliberately nasty: multi-byte characters, an emoji outside the BMP, repeated
# whitespace, and a tab. Plain ASCII would pass for every tokenizer ever written
# and tell us nothing.
PROBE = "Hello, world! café 中文 🙂\n\ttabs  and  spaces"

T = TypeVar("T", bound="Tokenizer")

_REGISTRY: dict[str, type[Tokenizer]] = {}


def register_tokenizer(name: str) -> Callable[[type[T]], type[T]]:
    """Register a tokenizer class under ``name``.

    Mirrors the mixer registry in ``litterbox.model.registry``. Kept local so
    that adding a tokenizer never requires touching model code.
    """

    def decorator(cls: type[T]) -> type[T]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"tokenizer {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_tokenizer(name: str) -> type[Tokenizer]:
    """Look up a registered tokenizer class."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown tokenizer {name!r}; available: {', '.join(available_tokenizers())}"
        ) from None


def available_tokenizers() -> list[str]:
    """Every registered tokenizer name, sorted."""
    return sorted(_REGISTRY)


def build_tokenizer(cfg: dict) -> Tokenizer:
    """Construct a tokenizer from a config block.

    ``{"type": "tiktoken", "encoding": "gpt2"}`` -> ``TiktokenTokenizer(encoding="gpt2")``
    """
    cfg = dict(cfg)
    return get_tokenizer(cfg.pop("type"))(**cfg)


class Tokenizer(ABC):
    """Text <-> token ids.

    Subclasses set the attributes below in ``__init__`` and then call
    ``self._validate()`` as the last statement. Validation is not optional: it
    is what turns "I called the right method" into "the backend demonstrably
    behaved".
    """

    name: str
    vocab_size: int
    bos_id: int | None = None
    eos_id: int | None = None
    is_lossless: bool = False

    # ---------------------------------------------------------------- core

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Text -> ids. Content only: no BOS, no EOS, no special tokens."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Ids -> text.

        Implementations must join all the bytes and decode once at the end. A
        multi-byte character can straddle two tokens, so decoding per token
        raises ``UnicodeDecodeError`` on any non-ASCII text.
        """

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        """Encode many strings.

        Correct but slow by default. Backends with a real batch path override
        this — on a multi-GB corpus the difference is minutes versus hours.
        """
        return [self.encode(t) for t in texts]

    # ----------------------------------------------------------- validation

    def _validate(self) -> None:
        """Prove the backend obeys the contract. Called once, at construction."""
        ids = self.encode(PROBE)

        if not ids:
            raise ValueError(f"{self.name}: encode() returned nothing for a non-empty probe")

        # Rule 1 — no special tokens crept in. This is the check that catches
        # HuggingFace silently prepending BOS despite add_special_tokens=False.
        if self.bos_id is not None and ids[0] == self.bos_id:
            raise ValueError(
                f"{self.name}: encode() prepended BOS ({self.bos_id}). "
                f"encode() must return content tokens only."
            )
        if self.eos_id is not None and ids[-1] == self.eos_id:
            raise ValueError(f"{self.name}: encode() appended EOS ({self.eos_id}).")

        # Rule 2 — every id fits the embedding matrix. Getting this wrong
        # surfaces as a CUDA index error partway through training, which looks
        # convincingly like a data bug.
        biggest = max(ids)
        if biggest >= self.vocab_size:
            raise ValueError(
                f"{self.name}: produced id {biggest} but vocab_size is {self.vocab_size}. "
                f"An embedding matrix of that size would be indexed out of range."
            )
        for label, tok_id in (("bos_id", self.bos_id), ("eos_id", self.eos_id)):
            if tok_id is not None and tok_id >= self.vocab_size:
                raise ValueError(
                    f"{self.name}: {label}={tok_id} is outside vocab_size {self.vocab_size}"
                )

        # Rule 3 — measured, never assumed.
        try:
            self.is_lossless = self.decode(ids) == PROBE
        except UnicodeDecodeError:
            self.is_lossless = False

    # ------------------------------------------------------------ niceties

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        lossy = "" if self.is_lossless else ", lossy"
        return f"<{type(self).__name__} {self.name!r} vocab={self.vocab_size:,}{lossy}>"
