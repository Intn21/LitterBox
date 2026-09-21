"""Greedy and sampled generation, driving every mixer through MixerState.

Generation code never branches on mixer type: whether a layer carries a KV
cache or a recurrent matrix is the mixer's business. It asks the model for
empty states, hands them back on every call, and advances one number —
``pos_offset``, the count of tokens processed so far.

Two generators live here, and they are meant to be read together.

:func:`generate_uncached` re-runs the whole sequence for every new token. It is
quadratic, wasteful, and impossible to get subtly wrong, because it is nothing
but the training-time forward pass called in a loop. That makes it the
**oracle**.

:func:`generate` does the work once. It runs the prompt through the model in a
single pass (the *prefill*), which fills every layer's state, then feeds tokens
back one at a time (the *decode*), each step computing only the new token. The
claim is that these produce exactly the same tokens, and
``tests/test_generation.py`` holds them to it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _pick_next(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> Tensor:
    """Choose the next token from ``[batch, vocab]`` logits. Shared by both
    generators, so they can differ only in how the logits were computed."""
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    # Sample on the CPU: a torch.Generator is tied to one device, and this
    # keeps a seed meaning the same text on every backend.
    return torch.multinomial(probs.cpu(), 1, generator=generator).to(logits.device)


@torch.no_grad()
def generate(
    model: nn.Module,
    ids: Tensor,
    max_new_tokens: int,
    *,
    max_context: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Extend ``ids`` using each mixer's inference state — one new token of work per step.

    Args:
        model: exposes ``init_states(batch, max_len)`` and
            ``forward(ids, states, pos_offset)``.
        ids: the prompt, ``[batch, seq]`` of int64, on the model's device.
        max_context: the model's ``max_seq_len``. The prompt plus the new tokens
            must fit. :func:`generate_uncached` can slide its window past this,
            because it recomputes everything anyway; a cache cannot, since the
            keys it holds were rotated at positions that would all have to shift.
        temperature: ``0`` is greedy (argmax). Above that, logits are divided
            by it before sampling: below 1 sharpens, above 1 flattens.
        top_k: keep only the ``k`` most likely tokens before sampling.
        eos_id: stop once every sequence in the batch has produced it.
        generator: seeds the sampling, for reproducible text.
    """
    batch, prompt_len = ids.shape
    total = prompt_len + max_new_tokens
    if total > max_context:
        raise ValueError(
            f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) = {total} exceeds "
            f"max_context={max_context}. A KV cache cannot slide: its keys were rotated at "
            f"absolute positions. Shorten the request, or use generate_uncached."
        )
    was_training = model.training
    model.eval()

    states = model.init_states(batch, total)
    # Prefill: the whole prompt in one parallel pass, filling every layer's state.
    # Only the last position's logits matter — they predict the first new token.
    logits = model(ids, states, pos_offset=0)[:, -1, :]

    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    for step in range(max_new_tokens):
        next_id = _pick_next(logits, temperature, top_k, generator)
        ids = torch.cat((ids, next_id), dim=1)
        if eos_id is not None:
            finished |= next_id.squeeze(1) == eos_id
            if bool(finished.all()):
                break
        if step + 1 == max_new_tokens:
            break  # the last token needs no forward pass: nothing will read its logits
        # Decode: feed back just the new token. Its position is the number of
        # tokens before it, which is also how many the caches now hold.
        logits = model(next_id, states, pos_offset=ids.shape[1] - 1)[:, -1, :]

    model.train(was_training)
    return ids


@torch.no_grad()
def generate_uncached(
    model: nn.Module,
    ids: Tensor,
    max_new_tokens: int,
    *,
    max_context: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Extend ``ids`` one token at a time, recomputing everything each step.

    Same arguments as :func:`generate`, with one difference: once the text
    outgrows ``max_context`` the oldest tokens are dropped and generation
    carries on, which for a rotary model also restarts positions at the new
    left edge.
    """
    was_training = model.training
    model.eval()
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        logits = model(ids[:, -max_context:])[:, -1, :]  # the last position predicts the next
        next_id = _pick_next(logits, temperature, top_k, generator)
        ids = torch.cat((ids, next_id), dim=1)
        if eos_id is not None:
            finished |= next_id.squeeze(1) == eos_id
            if bool(finished.all()):
                break
    model.train(was_training)
    return ids
