"""Tokenizer contract tests.

Every tokenizer, whoever wrote it, has to satisfy the same properties. These
tests are parameterized over whatever is installed, so adding a backend or an
algorithm of your own means adding one line to ``all_tokenizers()`` rather than
a new test file.
"""

from __future__ import annotations

import pytest

from litterbox.data.tokenizer import (
    Tokenizer,
    available_tokenizers,
    build_tokenizer,
    compare,
    register_tokenizer,
    show_segmentation,
)

# Deliberately awful. ASCII-only cases pass for everything and prove nothing.
NASTY = [
    "hello",
    "café",
    "中文测试",
    "🙂🙃",
    "tabs\tand\nnewlines",
    "  leading and trailing  ",
    "",
    "a" * 1000,
    "<|endoftext|>",  # a special-token string as literal content
]


def _maybe(cfg: dict) -> Tokenizer | None:
    """Build a tokenizer, or return None if its dependency is missing."""
    try:
        return build_tokenizer(cfg)
    except ImportError:
        return None


def all_tokenizers() -> list[Tokenizer]:
    """Every tokenizer available in this environment."""
    candidates = [
        {"type": "byte"},
        {"type": "char"},
        {"type": "tiktoken", "encoding": "gpt2"},
        {"type": "tiktoken", "encoding": "cl100k_base"},
    ]
    return [t for t in (_maybe(c) for c in candidates) if t is not None]


TOKENIZERS = all_tokenizers()
IDS = [t.name for t in TOKENIZERS]


# --------------------------------------------------------------- registry


def test_registry_lists_the_builtins():
    names = available_tokenizers()
    assert {"byte", "char", "tiktoken", "hf", "sentencepiece"} <= set(names)


def test_unknown_name_names_the_alternatives():
    with pytest.raises(KeyError, match="byte"):
        build_tokenizer({"type": "definitely_not_a_tokenizer"})


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register_tokenizer("byte")
        class Impostor(Tokenizer):
            pass


# --------------------------------------------------------------- contract


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_ids_fit_the_embedding_matrix(tok: Tokenizer):
    """Every id must be < vocab_size, or training dies with an index error."""
    for text in NASTY:
        ids = tok.encode(text)
        assert all(0 <= i < tok.vocab_size for i in ids), f"out-of-range id for {text!r}"


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_encode_adds_no_special_tokens(tok: Tokenizer):
    """encode() returns content only. A swap must never shift a dataset."""
    ids = tok.encode("hello world")
    if tok.bos_id is not None:
        assert ids[0] != tok.bos_id
    if tok.eos_id is not None:
        assert ids[-1] != tok.eos_id


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_encode_is_deterministic(tok: Tokenizer):
    for text in NASTY:
        assert tok.encode(text) == tok.encode(text)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_encode_batch_matches_encode(tok: Tokenizer):
    """The fast path must agree with the slow one."""
    assert tok.encode_batch(NASTY) == [tok.encode(t) for t in NASTY]


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_empty_string_encodes_to_nothing(tok: Tokenizer):
    assert tok.encode("") == []


@pytest.mark.parametrize("tok", [t for t in TOKENIZERS if t.is_lossless], ids=lambda t: t.name)
def test_lossless_tokenizers_round_trip(tok: Tokenizer):
    """A tokenizer claiming losslessness must survive everything in NASTY."""
    for text in NASTY:
        assert tok.decode(tok.encode(text)) == text, f"round-trip failed on {text!r}"


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_decode_joins_before_decoding(tok: Tokenizer):
    """Multi-byte characters may straddle tokens; decode must handle that.

    Decoding token-by-token raises UnicodeDecodeError on a continuation byte.
    Whole-sequence decode must not.
    """
    tok.decode(tok.encode("café 中文 🙂"))


# ------------------------------------------------------------- validation


def test_validate_catches_a_wrong_vocab_size():
    class LiesAboutVocab(Tokenizer):
        def __init__(self):
            self.name, self.vocab_size = "liar", 10
            self.bos_id = self.eos_id = None
            self._validate()

        def encode(self, text):
            return list(text.encode("utf-8"))

        def decode(self, ids):
            return bytes(ids).decode("utf-8")

    with pytest.raises(ValueError, match="vocab_size"):
        LiesAboutVocab()


def test_validate_catches_a_silently_prepended_bos():
    class PrependsBOS(Tokenizer):
        def __init__(self):
            self.name, self.vocab_size = "bos-adder", 300
            self.bos_id, self.eos_id = 1, None
            self._validate()

        def encode(self, text):
            return [1, *text.encode("utf-8")]

        def decode(self, ids):
            return bytes(ids[1:]).decode("utf-8")

    with pytest.raises(ValueError, match="prepended BOS"):
        PrependsBOS()


# ---------------------------------------------------------------- byte/char


def test_byte_tokenizer_is_exactly_one_token_per_byte():
    tok = build_tokenizer({"type": "byte"})
    for text in NASTY:
        assert len(tok.encode(text)) == len(text.encode("utf-8"))
    assert tok.vocab_size == 256
    assert tok.is_lossless


def test_char_tokenizer_is_lossy_outside_its_alphabet():
    """The failure mode that motivates working at the byte level."""
    tok = build_tokenizer({"type": "char"})
    assert not tok.is_lossless
    assert tok.decode(tok.encode("中文")) != "中文"


# ------------------------------------------------------------------ compare


def test_compare_ranks_by_compression():
    toks = [t for t in TOKENIZERS if t.name in {"byte", "tiktoken:gpt2"}]
    if len(toks) < 2:
        pytest.skip("needs both byte and tiktoken:gpt2")

    reports = compare(toks, "The quick brown fox. " * 100)
    assert reports[0].bytes_per_token > reports[-1].bytes_per_token
    assert reports[-1].name == "byte"  # one byte per token, by definition
    assert all(r.n_bytes == len("The quick brown fox. " * 100) for r in reports)


def test_show_segmentation_covers_the_whole_string():
    tok = build_tokenizer({"type": "byte"})
    assert show_segmentation(tok, "abc", sep="|") == "a|b|c"
