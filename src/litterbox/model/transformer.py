"""Backbone: embeddings, the stack of blocks, final norm, and LM head.

The stack is built by tiling ``layer_pattern`` to ``n_layers``, so a 3:1
hybrid and a dense baseline differ only in a config file. The backbone itself
never constructs a block — it receives them, the same way a block receives its
mixer. The only parameters it owns are the ones no block could: the token
embedding and the LM head, which under ``tie_embeddings`` are one matrix.

Two properties of that shared matrix are worth stating because tests rely on
them:

- **Lookup, not matmul.** The embedding's forward is a row gather, so its
  gradient is a scatter-add: rows of tokens absent from a batch receive
  exactly zero gradient. Padding rows (vocab rounded up to a multiple of 64
  for tensor-core alignment) therefore never train.
- **Small init keeps the start uniform.** With N(0, 0.02) rows, self-similarity
  ``e·e ≈ 0.02² · d_model`` is small enough that initial logits are near-flat
  and the first loss sits at ``ln(vocab_size)`` — the Step 1 exit check. A
  meaningfully lower first loss means the targets are leaking.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor

from litterbox.model.mixers.base import MixerState
from litterbox.positional import PositionalEncoding


class Transformer(nn.Module):
    """Decoder-only backbone around an injected stack of blocks.

    Args:
        vocab_size: rows in the embedding table. Padded vocab is fine; unused
            rows cost memory but never gradient.
        d_model: embedding width, and the width every block must preserve.
        blocks: the assembled ``TransformerBlock`` instances, in order. May be
            empty, which leaves embeddings → final norm → LM head: enough to
            test the ``ln(vocab_size)`` property before any mixer exists.
        pos: a :class:`~litterbox.positional.PositionalEncoding` whose additive
            ``embed`` hook runs here, right after token embedding. Rotary
            strategies leave that hook as identity and act inside attention
            layers instead, so passing RoPE here is valid and does nothing —
            which is the seam working, not a bug. ``None`` skips the call.
        final_norm: the norm applied after the last block. Injected like
            everything else. Defaults to identity, which keeps the blockless
            shell testable; a real model passes ``RMSNorm(d_model)``.
        tie_embeddings: reuse the embedding matrix as the LM head (GPT-2
            convention). Saves ``vocab_size × d_model`` parameters.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        blocks: Sequence[nn.Module] = (),
        pos: PositionalEncoding | None = None,
        final_norm: nn.Module | None = None,
        *,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = pos
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Init before tying: nn.Embedding defaults to N(0, 1), which would put
        # the first loss nowhere near ln(vocab_size).
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            # Same Parameter object, not a copy — one matrix, two roles. The
            # assignment must run this direction so the module keeps the
            # embedding's storage.
            self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        ids: Tensor,
        states: list[MixerState | None] | None = None,
        pos_offset: int = 0,
    ) -> Tensor:
        """Map token ids ``[batch, seq]`` (int64) to logits ``[batch, seq, vocab]``.

        Args:
            states: one inference state per block, from :meth:`init_states`, or
                ``None`` when training. The list is **updated in place** — each
                block's returned state is written back to its slot — so the
                return value stays a plain logits tensor in both modes.
            pos_offset: absolute position of ``ids[:, 0]``. Zero for a full
                sequence; when decoding, the number of tokens already processed.
                Reaches both positional hooks: ``embed`` here, ``rotate`` inside
                each attention layer.
        """
        if states is not None and len(states) != len(self.blocks):
            raise ValueError(f"got {len(states)} states for {len(self.blocks)} blocks")
        x = self.tok_emb(ids)
        if self.pos is not None:
            x = self.pos.embed(x, pos_offset)
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, None if states is None else states[i], pos_offset)
            if states is not None:
                states[i] = new_state
        x = self.final_norm(x)
        return self.lm_head(x)

    def init_states(self, batch_size: int, max_len: int) -> list[MixerState]:
        """One empty inference state per block, on this model's device and dtype.

        Each block answers for itself, so a hybrid's list can hold a KV cache in
        one slot and a recurrent matrix in the next without this method, or
        anything that uses it, knowing the difference.
        """
        ref = self.tok_emb.weight
        return [
            block.init_state(batch_size, max_len, dtype=ref.dtype, device=ref.device)
            for block in self.blocks
        ]

    @property
    def n_params(self) -> int:
        """Trainable parameters, counting tied weights once."""
        return sum(p.numel() for p in self.parameters())


def build_model(*args, **kwargs):
    """Instantiate a model from a validated model config."""
    raise NotImplementedError("Milestone 1: lands with utils/config.py and the first mixer.")
