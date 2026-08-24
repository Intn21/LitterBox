"""Wrappers around tokenizers other people trained.

Every import here is deferred into ``__init__``. Importing this module must not
require ``transformers``, which is a heavy dependency that most of the repo has
no use for.

Each wrapper's job is to translate one backend into the contract in ``base.py``.
The interesting lines are the ones that pin down behaviour the backend would
otherwise decide for itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from litterbox.data.tokenizer.base import Tokenizer, register_tokenizer


@register_tokenizer("tiktoken")
class TiktokenTokenizer(Tokenizer):
    """OpenAI's BPE tokenizers: GPT-2, GPT-4 (``cl100k_base``), GPT-4o (``o200k_base``).

    Fast, light, and lossless — byte-level BPE with no normalization.

    Note ``encode_ordinary`` rather than ``encode``. The plain ``encode`` method
    *raises* if the input contains a special-token string such as
    ``<|endoftext|>``; ``encode_ordinary`` treats it as ordinary text, which is
    both what we want and what makes the behaviour match the other backends.
    """

    def __init__(self, encoding: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("TiktokenTokenizer needs: pip install tiktoken") from e

        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"
        self.vocab_size = self._enc.n_vocab
        self.bos_id = None
        self.eos_id = self._enc.eot_token
        self._validate()

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return self._enc.encode_ordinary_batch(list(texts))


@register_tokenizer("hf")
class HFTokenizer(Tokenizer):
    """Any tokenizer on the HuggingFace Hub — Llama, Qwen, Mistral, Gemma, BERT.

    Two details here are load-bearing:

    ``add_special_tokens=False`` on every call. Without it, most instruct-tuned
    tokenizers silently prepend BOS, which shifts an entire dataset by one
    position with no error anywhere.

    ``len(self._tok)`` rather than ``self._tok.vocab_size``. The former counts
    added tokens, the latter does not, and the embedding matrix needs the
    former. Using the wrong one surfaces as an index error partway into
    training.

    Prefer ungated models for examples — ``Qwen/Qwen2.5-0.5B`` is Apache-2.0 and
    downloads without a login, whereas Llama and Gemma require accepting terms.
    """

    def __init__(self, model_id: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("HFTokenizer needs: pip install transformers") from e

        self._tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.name = f"hf:{model_id}"
        self.vocab_size = len(self._tok)
        self.bos_id = self._tok.bos_token_id
        self.eos_id = self._tok.eos_token_id
        self._validate()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        # clean_up_tokenization_spaces defaults to True in older versions and
        # rewrites spacing around punctuation, which makes a lossless tokenizer
        # look lossy. Turn it off so the round-trip measures the tokenizer.
        return self._tok.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return self._tok(list(texts), add_special_tokens=False)["input_ids"]


@register_tokenizer("sentencepiece")
class SentencePieceTokenizer(Tokenizer):
    """A raw SentencePiece ``.model`` file — Llama-2, T5, and older releases.

    Mostly useful when you have the model file but not a Hub repo; otherwise
    :class:`HFTokenizer` wraps the same thing with less ceremony.

    SentencePiece reports ``-1`` for unset special tokens rather than ``None``,
    which is normalized here.
    """

    def __init__(self, model_file: str | Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("SentencePieceTokenizer needs: pip install sentencepiece") from e

        self._sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.name = f"sentencepiece:{Path(model_file).stem}"
        self.vocab_size = self._sp.get_piece_size()
        self.bos_id = self._sp.bos_id() if self._sp.bos_id() >= 0 else None
        self.eos_id = self._sp.eos_id() if self._sp.eos_id() >= 0 else None
        self._validate()

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self._sp.decode(list(ids))

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return self._sp.encode(list(texts), out_type=int, add_bos=False, add_eos=False)
