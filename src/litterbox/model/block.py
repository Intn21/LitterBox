"""TransformerBlock — a token mixer and a channel mixer, each behind a norm, each
added back onto the residual stream.

    x = x + mixer(mixer_norm(x))      gather: move information between positions
    x = x + mlp(mlp_norm(x))          think:  process each position on its own

Everything is injected. The block receives instances and never constructs or
inspects them, which is what lets a hybrid give every layer a different mixer,
positional strategy, or MLP from config alone. The moment this file grows an
``isinstance`` check, the compartment has stopped being one.

**The residual stream.** Read the two lines above as: ``x`` is a running total
that every sub-layer *adds to* and none *replaces*. A sub-layer's output is a
correction, not a new representation. Two things follow. Gradients have a
direct path from the loss to every layer — the ``x +`` is an identity route no
sub-layer can block — which is what makes deep stacks trainable at all. And a
sub-layer that outputs zero is a no-op, so a freshly initialised deep model
starts out close to a shallow one and grows into its depth.

**Pre-norm.** The norm sits on the *branch*, in front of the sub-layer, not on
the stream. Each sub-layer reads a fixed-scale copy of ``x`` while the stream
itself is never rescaled, so the identity route stays clean. The original
Transformer normalised the stream after each addition (post-norm), which puts
a norm in the gradient's path at every layer and needs careful warmup to
train; pre-norm is what GPT-2 onward use. Its one cost: the stream's magnitude
grows with depth, which is why the backbone applies a final norm before the
LM head.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.model.mixers.base import MixerState, TokenMixer


class TransformerBlock(nn.Module):
    """One layer of the stack: pre-norm residual wiring around two injected halves.

    Args:
        mixer: the token mixer — the only part of the block allowed to move
            information between positions. Any :class:`TokenMixer`.
        mlp: the channel mixer, applied to each position independently. Any
            module mapping ``[batch, seq, d_model]`` to the same shape.
        mixer_norm: normalises what the mixer reads.
        mlp_norm: normalises what the MLP reads. Must be a different instance
            from ``mixer_norm``: passing one module twice would silently tie
            the two gains together, and the two sub-layers want different ones.
    """

    def __init__(
        self,
        mixer: TokenMixer,
        mlp: nn.Module,
        mixer_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        if mixer_norm is mlp_norm:
            raise ValueError(
                "mixer_norm and mlp_norm are the same module instance, which would tie "
                "their learned gains; construct one norm per sub-layer"
            )
        self.mixer_norm = mixer_norm
        self.mixer = mixer
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(
        self,
        x: Tensor,
        state: MixerState | None = None,
        pos_offset: int = 0,
    ) -> tuple[Tensor, MixerState | None]:
        """Map ``[batch, seq, d_model]`` to the same shape, threading mixer state.

        ``state`` and ``pos_offset`` belong to the mixer and are passed straight
        through; the block has no opinion on either. Only the mixer carries
        inference state — the norms and the MLP are per-token and stateless.
        """
        mixed, state = self.mixer(self.mixer_norm(x), state, pos_offset)
        x = x + mixed
        x = x + self.mlp(self.mlp_norm(x))
        return x, state

    def init_state(self, batch_size: int, max_len: int, **kwargs) -> MixerState:
        """The mixer's empty inference state. The block adds nothing to it — the
        norms and the MLP are per-token, so there is nothing else to remember."""
        return self.mixer.init_state(batch_size, max_len, **kwargs)


def scale_residual_projections(blocks: Iterable[nn.Module], *, base_std: float = 0.02) -> int:
    """Re-initialise every residual branch's output projection for the stack's depth.

    Each block adds two branch outputs onto the stream, so after ``n`` blocks
    the stream is a sum of ``2n`` contributions. If each has variance ``σ²``,
    the sum has variance ``2n·σ²`` — the stream grows with depth, and a deeper
    model starts training from a louder, less stable place than a shallow one.
    GPT-2's fix is to shrink the *last* projection of every branch by
    ``1/sqrt(2n)``, so the summed variance is the same at any depth.

    This lives outside the block because a block cannot know how deep the
    stack it sits in is. And it finds the projections without inspecting any
    mixer or MLP: a branch opts in by setting ``residual_out = True`` on its
    final ``nn.Linear`` (attention's ``o_proj``, SwiGLU's ``down_proj``). A new
    mixer that wants the scaling sets the same flag; nothing here changes.

    Returns:
        The number of projections rescaled — ``2 · len(blocks)`` when every
        branch opted in, which is worth asserting at the call site.
    """
    blocks = list(blocks)
    if not blocks:
        return 0
    std = base_std / math.sqrt(2 * len(blocks))
    count = 0
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.Linear) and getattr(module, "residual_out", False):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                count += 1
    return count


def build_block(*args, **kwargs):
    """Assemble one block from a ``layer_pattern`` entry and the model config."""
    raise NotImplementedError("Lands with utils/config.py, which resolves names to instances.")
