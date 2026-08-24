"""How the tokenizer wrapper works.

Run it::

    python examples/tokenizers.py

Everything degrades gracefully: with nothing optional installed you still get
the byte and char backends. ``pip install -e ".[tokenizers]"`` adds GPT-2/4/4o
and anything on the HuggingFace Hub.

The last section is the point of the whole interface — plugging in an algorithm
of your own and measuring it against the others without touching any other file.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from litterbox.data.tokenizer import (
    Tokenizer,
    available_tokenizers,
    build_tokenizer,
    compare_segmentation,
    print_comparison,
    register_tokenizer,
)

CORPUS = (
    "Once upon a time there was a little girl who lived near a river. "
    "Every morning she walked to the river to fetch water for her family. "
    "The water was cold and clear and the stones shone like little stars. "
    "One day she found a small silver key between the stones by the river. "
) * 60

HELD_OUT = (
    "One morning the little girl walked to the river and found a silver key. "
    "She showed the key to her grandmother, who looked at it for a long time. "
)

CODE = "def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)"


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * len(title))


def build(cfg: dict) -> Tokenizer | None:
    """Build a tokenizer, or explain why we can't and move on."""
    try:
        return build_tokenizer(cfg)
    except ImportError as e:
        print(f"  skipping {cfg['type']}: {e}")
        return None


# ═══════════════════════════════════════════════════ bring your own algorithm


@register_tokenizer("word")
class WordTokenizer(Tokenizer):
    """A whole-word vocabulary with a byte-level fallback.

    Not a good tokenizer — that is rather the point. It compresses in-domain
    text beautifully and collapses on anything it has not seen, which is the
    behaviour the comparison harness exists to make visible.

    The byte fallback is what keeps it lossless: an unknown chunk is emitted as
    its raw bytes rather than an unknown token, so nothing is ever destroyed.
    """

    # A chunk is a word with its leading space, or a run of whitespace. Every
    # character matches one of the two, so the split covers the input exactly —
    # which is what makes the round-trip work.
    PATTERN = re.compile(r" ?[^\s]+|\s+")

    def __init__(self, chunks: Sequence[str]) -> None:
        self.itos: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.stoi: dict[str, int] = {}
        for n, chunk in enumerate(chunks):
            self.itos[256 + n] = chunk.encode("utf-8")
            self.stoi[chunk] = 256 + n

        self.name = f"word[{len(chunks)}]"
        self.vocab_size = 256 + len(chunks)
        self.bos_id = None
        self.eos_id = None
        self._validate()

    @classmethod
    def train(cls, text: str, vocab_size: int = 4096) -> WordTokenizer:
        counts = Counter(cls.PATTERN.findall(text))
        keep = [c for c, _ in counts.most_common(max(0, vocab_size - 256))]
        return cls(keep)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self.PATTERN.findall(text):
            known = self.stoi.get(chunk)
            if known is None:
                ids.extend(chunk.encode("utf-8"))  # fall back to bytes
            else:
                ids.append(known)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # Join every piece, then decode once. Decoding per id would fail the
        # moment a multi-byte character got split across two of them.
        return b"".join(self.itos[i] for i in ids).decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════ main


def main() -> None:
    section("1. What's registered")
    print(f"  {', '.join(available_tokenizers())}")
    print("\n  Anything decorated with @register_tokenizer shows up here,")
    print("  including 'word' further down this file.")

    section("2. Building one, and the round trip")
    tok = build_tokenizer({"type": "byte"})
    print(f"  {tok!r}")

    text = "café 🙂"
    ids = tok.encode(text)
    print(f"\n  encode({text!r})")
    print(f"    -> {ids}")
    print(f"  decode(...) -> {tok.decode(ids)!r}")
    print(f"\n  {len(text)} characters became {len(ids)} tokens — 'é' and '🙂'")
    print("  are multi-byte, so a character is not a token.")

    section("3. Special tokens are data, not behaviour")
    gpt2 = build({"type": "tiktoken", "encoding": "gpt2"})
    if gpt2:
        print(f"  {gpt2.name}: bos_id={gpt2.bos_id}  eos_id={gpt2.eos_id}")
        print(f"\n  encode('hello') -> {gpt2.encode('hello')}")
        print("  Content only. encode() never adds BOS or EOS, on any backend,")
        print("  so swapping tokenizers cannot silently shift your dataset.")
        print("\n  Want them? Add them yourself, where it's visible:")
        print("    ids = [tok.bos_id, *tok.encode(doc), tok.eos_id]")

    section("4. Which one compresses best?")
    tokenizers = [
        t
        for t in (
            build_tokenizer({"type": "byte"}),
            build_tokenizer({"type": "char"}),
            build({"type": "tiktoken", "encoding": "gpt2"}),
            build({"type": "tiktoken", "encoding": "o200k_base"}),
        )
        if t is not None
    ]

    print()
    print_comparison(tokenizers, HELD_OUT + CODE)
    print("\n  bytes/token is the number that matters: how much text fits in")
    print("  one model position. 'byte' is 1.00 by definition — the floor.")

    section("5. Where do they actually split?")
    real = [t for t in tokenizers if t.name.startswith("tiktoken")]
    if real:
        print()
        print(compare_segmentation(real, CODE))
        print("\n  Identical token count, different choices. o200k keeps '(n'")
        print("  as one token where gpt2 splits it — but gpt2 keeps ' 2'")
        print("  together where o200k separates the space from the digit.")
        print("  Newer tokenizers isolate digits deliberately: it costs a")
        print("  little compression and helps the model do arithmetic.")

    section("6. Plug in your own")
    print("  WordTokenizer is defined in this file — one class, one decorator,")
    print("  no other file touched. Training it on the sample corpus:\n")

    word = WordTokenizer.train(CORPUS, vocab_size=1024)
    print(f"  {word!r}")

    print("\n  On text like what it was trained on:")
    print()
    print_comparison([*tokenizers, word], HELD_OUT)

    print("\n  On Python source, which it has never seen:")
    print()
    print_comparison([*tokenizers, word], CODE)

    print("\n  That collapse is the lesson. A tokenizer is only as good as the")
    print("  match between its training data and what you actually feed it —")
    print("  and this harness is how you find that out in seconds rather than")
    print("  after a training run.")

    section("7. The contract is enforced, not assumed")
    print("  Every tokenizer self-checks at construction. Here's one that lies:\n")

    class Broken(Tokenizer):
        def __init__(self):
            self.name, self.vocab_size = "broken", 10
            self.bos_id = self.eos_id = None
            self._validate()

        def encode(self, text):
            return list(text.encode("utf-8"))

        def decode(self, ids):
            return bytes(ids).decode("utf-8")

    try:
        Broken()
    except ValueError as e:
        print(f"  ValueError: {e}")

    print("\n  Caught at construction, in milliseconds — rather than as a CUDA")
    print("  index error four hours into training.")


if __name__ == "__main__":
    main()
