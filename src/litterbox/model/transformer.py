"""The backbone: embeddings, a stack of blocks, a final norm, and an LM head.

The stack is built by tiling ``layer_pattern`` up to ``n_layers``, so a dense
baseline and a 3:1 hybrid differ only in a config file. Everything here operates
on each token independently — the only place positions interact is inside a
token mixer.

Parameter names are namespaced by compartment (``blocks.3.token_mixer.*``,
``blocks.3.channel_mixer.*``) so that a checkpoint trained with one mixer can
later be partially loaded into a model whose layer 3 is something else. That
costs nothing now and is miserable to retrofit once checkpoints exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from litterbox.model import channel_mixers as _channel_mixers  # noqa: F401  (registration)
from litterbox.model import mixers as _mixers  # noqa: F401  (registration)
from litterbox.model import norm as _norm  # noqa: F401  (registration)
from litterbox.model.block import TransformerBlock
from litterbox.model.channel_mixers.base import build_channel_mixer
from litterbox.model.norm import build_norm
from litterbox.model.registry import get_mixer

if TYPE_CHECKING:
    from torch import Tensor

    from litterbox.model.mixers.base import MixerState


def tile_layer_pattern(pattern: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Repeat ``pattern`` up to ``n_layers``.

    Raises:
        ValueError: if the pattern does not divide evenly. A 5-layer pattern in a
            12-layer model is almost always a typo, and silently truncating it
            would produce a model nobody intended.
    """
    if not pattern:
        raise ValueError("layer_pattern is empty")
    if n_layers % len(pattern) != 0:
        raise ValueError(
            f"layer_pattern of length {len(pattern)} does not tile evenly into n_layers={n_layers}"
        )
    return [dict(pattern[i % len(pattern)]) for i in range(n_layers)]


class Transformer(nn.Module):
    """A decoder-only transformer with swappable compartments."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        layer_pattern: list[dict[str, Any]],
        mlp: dict[str, Any] | None = None,
        norm: dict[str, Any] | None = None,
        max_seq_len: int = 2048,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()

        mlp = dict(mlp or {"type": "swiglu", "hidden_mult": 4})
        norm = dict(norm or {"type": "rmsnorm", "eps": 1e-5})

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)

        specs = tile_layer_pattern(layer_pattern, n_layers)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                token_mixer=self._build_mixer(spec, d_model, max_seq_len),
                channel_mixer=build_channel_mixer(d_model, mlp),
                norm_1=build_norm(d_model, **norm),
                norm_2=build_norm(d_model, **norm),
            )
            for spec in specs
        )

        self.norm_f = build_norm(d_model, **norm)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

        # Scale the residual-path output projections by depth, so the residual
        # stream does not grow without bound as layers accumulate (GPT-2's trick).
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _build_mixer(spec: dict[str, Any], d_model: int, max_seq_len: int):
        spec = dict(spec)
        name = spec.pop("mixer")
        spec.setdefault("max_seq_len", max_seq_len)
        return get_mixer(name)(d_model, **spec)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        idx: Tensor,
        states: list[MixerState | None] | None = None,
        pos_offset: int = 0,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, list[MixerState | None] | None, Tensor | None]:
        """Run the stack.

        Args:
            idx: ``[B, S]`` token ids.
            states: per-layer inference state, or ``None`` during training.
            pos_offset: absolute position of ``idx[:, 0]``.
            targets: ``[B, S]`` next-token labels. When given, the loss is
                returned alongside the logits.

        Returns:
            ``(logits, states, loss)``.
        """
        x = self.embed(idx)

        if states is None:
            states = [None] * len(self.blocks)
        new_states: list[MixerState | None] = []

        for block, state in zip(self.blocks, states, strict=True):
            x, state = block(x, state, pos_offset)
            new_states.append(state)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )

        out_states = new_states if any(s is not None for s in new_states) else None
        return logits, out_states, loss

    # ------------------------------------------------------------------ state

    def init_states(
        self,
        batch: int,
        max_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> list[MixerState]:
        """Allocate inference state for every layer.

        Generation calls this once and then threads the result through. It never
        learns whether a given layer holds a growing cache or a fixed matrix.
        """
        device = device or next(self.parameters()).device
        return [
            block.token_mixer.init_state(batch, max_len, device=device, dtype=dtype)
            for block in self.blocks
        ]

    def num_parameters(self, *, embeddings: bool = True) -> int:
        """Total parameters. Tied weights are counted once."""
        seen, total = set(), 0
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if not embeddings and "embed" in name:
                continue
            total += p.numel()
        return total


def build_model(cfg: dict[str, Any]) -> Transformer:
    """Instantiate a model from a validated model config."""
    return Transformer(**cfg)
