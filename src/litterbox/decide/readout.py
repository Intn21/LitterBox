"""The baseline readout: a request as lettered prompts, one pass, letters' logits.

This is the zero-training way to get a decision out of a causal language
model, and the baseline every later readout is compared against. Each question
is rendered as text whose natural continuation is a single letter::

    State:
    I was charged twice this month.

    Question: Which department should handle this?
    A = billing: payments, invoices
    B = technical: bugs, outages
    C = other
    Answer:

The model runs once per question, the logits at the last position are taken,
the entries for ``A``, ``B``, ``C`` are kept and the rest discarded, and a
softmax over those three is the answer. Nothing is trained; the model's
pretrained habit of continuing a lettered list is pointed at the question.

Two things are fixed by the tokenizer rather than by us. The letters must be
single tokens *in that position*: many tokenizers encode ``A`` and `` A``
differently, so :func:`label_ids` tries both after ``Answer:`` and keeps
whichever is one token. And the alphabet is finite, 62 labels, so this readout
refuses questions with more options than it has single-token labels. The
pointer head in step 2 removes that limit; this one is honest about it.

The model is anything that maps token ids ``[batch, seq]`` to logits
``[batch, seq, vocab]``: the repo's :class:`~litterbox.model.Transformer`
directly, or a ``transformers`` causal LM through :func:`hf_logits`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from litterbox.data.tokenizer import Tokenizer
from litterbox.decide.derive import respond
from litterbox.decide.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Request,
    Response,
    ScoreQuestion,
    Text,
)

# Letters in the order they are handed out. 62 is the ceiling of this readout.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

PREAMBLE = (
    "You will be shown a state and a question about it. "
    "Answer with the single letter of the best option."
)

# ids [batch, seq] and each row's answer position [batch] -> logits at those positions
# [batch, vocab]. Only the answer positions are needed, and a full [batch, seq, vocab]
# tensor over a 150k vocabulary is gigabytes per batch, so the adapters below gather first.
Logits = Callable[[Tensor, Tensor], Tensor]


# --- rendering ------------------------------------------------------------------


def _text(value: Text) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)


def options_of(question: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> list[tuple[str, str]]:
    """``(name, description)`` per answer, in the order the probabilities will come out."""
    if isinstance(question, ChoiceQuestion):
        return [(name, _text(desc)) for name, desc in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return [(f"level {i}", _text(desc)) for i, desc in enumerate(question.criteria)]
    crit = question.criteria or {}
    return [("no", _text(crit.get("false", ""))), ("yes", _text(crit.get("true", "")))]


def render(
    request: Request, qid: str, order: Sequence[int] | None = None, *, answer_line: bool = True
) -> str:
    """The prompt for one question of ``request``.

    ``order`` lists the options by index in the order to show them, for
    measuring order bias; the default is the request's order. The prompt ends
    on ``Answer:`` so the next token is the letter; a chat wrapper passes
    ``answer_line=False`` and supplies that line itself, inside the assistant
    turn (see :func:`chat_wrap`).
    """
    question = request.questions[qid]
    options = options_of(question)
    if order is None:
        order = range(len(options))
    if len(options) > len(ALPHABET):
        raise ValueError(
            f"{qid!r} has {len(options)} options; this readout has {len(ALPHABET)} labels. "
            "Use the pointer readout for wider questions."
        )
    lines = [
        PREAMBLE,
        "",
        "State:",
        _text(request.state),
        "",
        f"Question: {_text(question.instructions)}",
    ]
    for letter, j in zip(ALPHABET, order, strict=False):
        name, desc = options[j]
        lines.append(f"{letter} = {name}: {desc}" if desc else f"{letter} = {name}")
    if answer_line:
        lines.append("Answer:")
    return "\n".join(lines)


Wrap = Callable[[str], str]  # prompt without its Answer: line -> the text the model sees


def chat_wrap(hf_tokenizer, **template_kwargs) -> Wrap:
    """A :data:`Wrap` that frames the prompt as a chat turn and opens the answer.

    Chat checkpoints answer worse outside their template. The prompt becomes
    the user message, the assistant turn is opened, and ``Answer:`` is written
    as the start of the assistant's reply, so the next token is the spaced
    letter exactly as in the bare prompt. ``template_kwargs`` reach
    ``apply_chat_template``: Qwen3-family models take ``enable_thinking=False``.
    """

    def wrap(prompt: str) -> str:
        framed = hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return framed + "Answer:"

    return wrap


# --- tokens ---------------------------------------------------------------------


def label_ids(tokenizer: Tokenizer, n: int) -> list[int]:
    """Token ids of the first ``n`` labels, each a single token in the answer position.

    Tries `` A`` (with the space a tokenizer merges after ``Answer:``) and then
    ``A``. The whole alphabet must agree on one form, so that a question's
    options are compared on equal terms.
    """
    if n > len(ALPHABET):
        raise ValueError(f"{n} labels requested, {len(ALPHABET)} available")
    for prefix in (" ", ""):
        ids = [tokenizer.encode(prefix + c) for c in ALPHABET[:n]]
        if all(len(t) == 1 for t in ids):
            return [t[0] for t in ids]
    raise ValueError(f"{tokenizer.name}: the labels are not single tokens, with or without a space")


def encode_prompts(
    tokenizer: Tokenizer, prompts: Sequence[str], pad_id: int = 0
) -> tuple[Tensor, Tensor]:
    """Right-padded ids ``[n, max_len]`` and each prompt's last real position ``[n]``.

    Right padding is safe for a causal model: nothing before a prompt's last
    token can see the padding after it, so the logits there are exactly what
    the unpadded prompt would give.
    """
    encoded = [tokenizer.encode(p) for p in prompts]
    width = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        ids[i, : len(e)] = torch.tensor(e)
    last = torch.tensor([len(e) - 1 for e in encoded])
    return ids, last


# --- the readout ------------------------------------------------------------


@torch.no_grad()
def batched_probabilities(
    logits_fn: Logits,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    widths: Sequence[int],
    *,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> tuple[list[list[float]], int]:
    """One probability vector per prompt, over that prompt's first ``widths[i]`` labels.

    This is the readout proper, independent of the request schema, so that a
    dataset of many states can be run as one stream of prompts. Returns the
    vectors and the total number of prompt tokens.
    """
    labels = torch.tensor(label_ids(tokenizer, max(widths)))
    out: list[list[float]] = []
    total_tokens = 0
    for start in range(0, len(prompts), batch_size):
        chunk = slice(start, start + batch_size)
        ids, last = encode_prompts(tokenizer, prompts[chunk])
        total_tokens += int(last.sum()) + len(last)
        if device is not None:
            ids = ids.to(device)
        at_last = logits_fn(ids, last.to(ids.device))  # [b, vocab], at each prompt's Answer:
        scores = at_last[:, labels.to(at_last.device)].float().cpu()  # [b, max_width]
        for row, k in zip(scores, widths[chunk], strict=True):
            out.append(row[:k].softmax(0).tolist())  # only this prompt's k labels exist
    return out, total_tokens


def probabilities(
    logits_fn: Logits,
    tokenizer: Tokenizer,
    request: Request,
    *,
    orders: dict[str, Sequence[int]] | None = None,
    wrap: Wrap | None = None,
    batch_size: int = 16,
    device: torch.device | str | None = None,
) -> tuple[dict[str, list[float]], int]:
    """One probability vector per question id, and the total prompt tokens.

    Each vector is in the request's option order, whatever ``orders`` showed
    the model, so that callers never see the permutation. ``wrap`` frames each
    prompt for a chat model (:func:`chat_wrap`); without it the bare prompt is
    used.
    """
    orders = orders or {}
    qids = list(request.questions)
    if wrap is None:
        prompts = [render(request, qid, orders.get(qid)) for qid in qids]
    else:
        prompts = [wrap(render(request, qid, orders.get(qid), answer_line=False)) for qid in qids]
    widths = [len(options_of(request.questions[qid])) for qid in qids]
    rows, total_tokens = batched_probabilities(
        logits_fn, tokenizer, prompts, widths, batch_size=batch_size, device=device
    )
    out: dict[str, list[float]] = {}
    for qid, p in zip(qids, rows, strict=True):
        shown = orders.get(qid)
        if shown is not None:  # map back from the shown order to the request's
            unshuffled = [0.0] * len(p)
            for slot, j in enumerate(shown):
                unshuffled[j] = p[slot]
            p = unshuffled
        out[qid] = p
    return out, total_tokens


def decide(
    logits_fn: Logits,
    tokenizer: Tokenizer,
    request: Request,
    *,
    model_name: str = "litterbox-readout",
    **kw,
) -> Response:
    """Answer ``request`` with the baseline readout, as a checked :class:`Response`."""
    probs, tokens = probabilities(logits_fn, tokenizer, request, **kw)
    return respond(request, probs, model=model_name, input_tokens=tokens)


def full_logits(model) -> Logits:
    """Adapt a model that returns logits for every position, like the repo's backbone."""

    def fn(ids: Tensor, last: Tensor) -> Tensor:
        logits = model(ids)  # [b, seq, vocab]
        return logits[torch.arange(len(last), device=logits.device), last.to(logits.device)]

    return fn


def hf_logits(model) -> Logits:
    """Adapt a ``transformers`` causal LM, applying its output head only at the answer positions.

    The base model's hidden states are gathered at ``last`` before the output
    embedding is applied, so memory scales with the hidden size rather than
    the vocabulary. The attention mask marks nothing as padding: padding is on
    the right and never read, and every row sees plain positions ``0..n-1``.
    """
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()

    def fn(ids: Tensor, last: Tensor) -> Tensor:
        device = next(model.parameters()).device
        ids = ids.to(device)
        hidden = base(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        at_last = hidden[torch.arange(len(last), device=device), last.to(device)]  # [b, d]
        return head(at_last)  # [b, vocab]

    return fn
