"""Assemble a dense model from plain arguments.

A convenience over :func:`litterbox.model.build.build_model` for the common case
where every layer is the same: it writes the one-entry ``layer_pattern`` for you.
There is a single construction path either way — this builds a
:class:`ModelConfig` and hands it over — so a model made here is exactly the
model the equivalent YAML would make.
"""

from __future__ import annotations

from litterbox.model.build import build_model
from litterbox.model.transformer import Transformer
from litterbox.utils.config import ModelConfig


def dense_transformer(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    heads: int,
    kv_heads: int | None = None,
    *,
    max_seq_len: int = 2048,
    mixer: str = "full_attention",
    rope_layout: str = "half",
    rope_base: float = 10000.0,
    hidden_mult: float = 4.0,
    norm_eps: float = 1e-5,
    tie_embeddings: bool = True,
    scale_init: bool = True,
    **mixer_args,
) -> Transformer:
    """A Llama-shaped dense decoder: RoPE, grouped-query attention, SwiGLU, RMSNorm.

    Args:
        mixer: registry name of the token mixer used in every layer. Swapping
            ``"full_attention"`` for a fast-tier twin changes speed and memory,
            never the parameters or the numbers.
        max_seq_len: longest sequence RoPE has angles for. Reading past it
            raises rather than extrapolating.
        scale_init: apply GPT-2's depth-scaled init to every residual branch's
            final projection.
        **mixer_args: passed to every layer's mixer — ``window=64`` for
            ``mixer="sliding_window"``.
    """
    layer = {"mixer": mixer, "heads": heads, "pos": "rope", **mixer_args}
    if kv_heads is not None:
        layer["kv_heads"] = kv_heads
    return build_model(
        ModelConfig(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            layer_pattern=[layer],
            mlp={"hidden_mult": hidden_mult},
            norm={"eps": norm_eps},
            rope={"layout": rope_layout, "base": rope_base},
            tie_embeddings=tie_embeddings,
            scale_residual_init=scale_init,
        )
    )
