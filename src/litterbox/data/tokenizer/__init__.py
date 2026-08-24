"""Tokenizers: text <-> token ids, behind one interface.

Every backend obeys the same contract — ``encode`` returns content tokens only,
``vocab_size`` is what the embedding matrix needs, and losslessness is measured
rather than assumed. See ``base.py`` for why those rules exist.

Adding an algorithm of your own means writing a class and decorating it; nothing
else in the codebase needs to learn about it::

    @register_tokenizer("my_bpe")
    class MyBPE(Tokenizer):
        ...

Comparing it against the others is then one call::

    from litterbox.data.tokenizer import build_tokenizer, print_comparison

    toks = [build_tokenizer({"type": "byte"}),
            build_tokenizer({"type": "tiktoken", "encoding": "gpt2"}),
            MyBPE.train(corpus, vocab_size=8192)]
    print_comparison(toks, held_out_text)
"""

from litterbox.data.tokenizer import byte, external  # noqa: F401  (registration)
from litterbox.data.tokenizer.base import (
    PROBE,
    Tokenizer,
    available_tokenizers,
    build_tokenizer,
    get_tokenizer,
    register_tokenizer,
)
from litterbox.data.tokenizer.compare import (
    TokenizerReport,
    compare,
    compare_segmentation,
    format_comparison,
    measure,
    print_comparison,
    show_segmentation,
)

__all__ = [
    "PROBE",
    "Tokenizer",
    "TokenizerReport",
    "available_tokenizers",
    "build_tokenizer",
    "compare",
    "compare_segmentation",
    "format_comparison",
    "get_tokenizer",
    "measure",
    "print_comparison",
    "register_tokenizer",
    "show_segmentation",
]
