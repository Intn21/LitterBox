"""Assemble a model from plain arguments — the builder, before there are configs.

``utils/config.py`` will eventually turn a YAML ``layer_pattern`` into a model.
Underneath, that builder has to do exactly this: construct one positional
strategy, mixer, MLP and pair of norms per layer, inject them into a block, and
inject the blocks into the backbone. Spelling it out as a function means the
training loop and the tests share one construction, and the config builder has
something concrete to call when it lands.
"""

from __future__ import annotations

from litterbox.model.block import TransformerBlock, scale_residual_projections
from litterbox.model.mlp import SwiGLU
from litterbox.model.norm import RMSNorm
from litterbox.model.registry import get_mixer
from litterbox.model.transformer import Transformer
from litterbox.positional import RoPE


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
    """
    if d_model % heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by heads={heads}")
    mixer_cls = get_mixer(mixer)
    head_dim = d_model // heads
    blocks = [
        TransformerBlock(
            mixer=mixer_cls(
                d_model,
                heads,
                kv_heads,
                # One strategy instance per layer, as a per-layer config would give.
                pos=RoPE(head_dim, max_seq_len, layout=rope_layout, base=rope_base),
            ),
            mlp=SwiGLU(d_model, hidden_mult),
            mixer_norm=RMSNorm(d_model, norm_eps),
            mlp_norm=RMSNorm(d_model, norm_eps),
        )
        for _ in range(n_layers)
    ]
    if scale_init:
        rescaled = scale_residual_projections(blocks)
        assert rescaled == 2 * n_layers, "every residual branch should opt in to depth scaling"
    return Transformer(
        vocab_size,
        d_model,
        blocks,
        final_norm=RMSNorm(d_model, norm_eps),
        tie_embeddings=tie_embeddings,
    )
