"""Greedy and sampled generation, driving every mixer through MixerState.

Generation code never branches on mixer type: whether a layer carries a KV
cache or a recurrent matrix is the mixer's business.

Two generators live here. :func:`generate_uncached` re-runs the whole sequence
for every new token — quadratic, wasteful, and impossible to get subtly wrong,
because it is nothing but the training-time forward pass called in a loop. That
makes it the **oracle**: ROADMAP step 3's exit condition is that the cached
:func:`generate` produces exactly the same tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


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

    Args:
        model: maps token ids ``[batch, seq]`` to logits ``[batch, seq, vocab]``.
        ids: the prompt, ``[batch, seq]`` of int64, on the model's device.
        max_context: longest window fed to the model — its ``max_seq_len``.
            Once the text outgrows it the oldest tokens are dropped, which for
            a rotary model also restarts positions at the new left edge.
        temperature: ``0`` is greedy (argmax). Above that, logits are divided
            by it before sampling: below 1 sharpens, above 1 flattens.
        top_k: keep only the ``k`` most likely tokens before sampling. Cuts off
            the long tail of individually unlikely, collectively probable junk.
        eos_id: stop once every sequence in the batch has produced it.
        generator: seeds the sampling, for reproducible text.
    """
    was_training = model.training
    model.eval()
    finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        window = ids[:, -max_context:]
        logits = model(window)[:, -1, :].float()  # only the last position predicts the next token
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = logits.softmax(dim=-1)
            # Sample on the CPU: a torch.Generator is tied to one device, and
            # this keeps a seed meaning the same text on every backend.
            next_id = torch.multinomial(probs.cpu(), 1, generator=generator).to(ids.device)
        ids = torch.cat((ids, next_id), dim=1)
        if eos_id is not None:
            finished |= next_id.squeeze(1) == eos_id
            if bool(finished.all()):
                break
    model.train(was_training)
    return ids


def generate(*args, **kwargs):
    """Generate tokens from a prompt, threading state through the stack."""
    raise NotImplementedError("ROADMAP step 3: needs the KV cache in infer/cache.py.")
