"""Greedy and sampled generation.

This file is the practical test of whether the mixer abstraction is real. It
allocates state, threads it through the stack, and advances ``pos_offset`` — and
it never branches on what kind of mixer any layer holds. If a new technique
required a change here, the interface would be leaking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from litterbox.model.transformer import Transformer


@torch.no_grad()
def generate(
    model: Transformer,
    idx: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_cache: bool = True,
) -> Tensor:
    """Continue ``idx`` for ``max_new_tokens`` steps.

    Args:
        model: the transformer.
        idx: ``[B, S]`` prompt token ids.
        max_new_tokens: how many tokens to append.
        temperature: 0 means greedy (argmax); higher is more random.
        top_k: restrict sampling to the k most likely tokens.
        use_cache: when False, re-runs the whole prefix every step. Slow, but the
            two paths must agree exactly — that equivalence is the strongest
            single check that a mixer's state handling is correct, and
            ``tests/test_state_consistency.py`` asserts it.

    Returns:
        ``[B, S + max_new_tokens]``.
    """
    model.eval()
    B, S = idx.shape

    if use_cache:
        states = model.init_states(
            B, S + max_new_tokens, device=idx.device, dtype=next(model.parameters()).dtype
        )
        # Prefill: run the prompt once, then feed one token at a time.
        logits, states, _ = model(idx, states, pos_offset=0)
    else:
        states = None
        logits, _, _ = model(idx, None, pos_offset=0)

    for step in range(max_new_tokens):
        next_token = _sample(logits[:, -1, :], temperature=temperature, top_k=top_k)
        idx = torch.cat([idx, next_token], dim=1)

        if step == max_new_tokens - 1:
            break

        if use_cache:
            logits, states, _ = model(next_token, states, pos_offset=idx.shape[1] - 1)
        else:
            logits, _, _ = model(idx, None, pos_offset=0)

    return idx


def _sample(logits: Tensor, *, temperature: float, top_k: int | None) -> Tensor:
    """Pick one token per batch row from ``[B, vocab]`` logits."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
