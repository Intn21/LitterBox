"""Shared fixtures."""

import pytest

from litterbox.model import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    get_mixer,
    scale_residual_projections,
)
from litterbox.model.transformer import Transformer
from litterbox.positional import RoPE


@pytest.fixture
def assemble():
    """Build a small dense model by hand — no config system involved.

    This is the assembly a builder will eventually do from YAML, spelled out:
    one positional strategy, mixer, MLP, and pair of norms per layer, all
    injected into a block, the blocks injected into the backbone.
    """

    def _assemble(
        vocab=256, d_model=64, heads=4, kv_heads=2, n_layers=2, max_seq_len=128, scale_init=True
    ):
        attention = get_mixer("full_attention")
        blocks = [
            TransformerBlock(
                mixer=attention(
                    d_model, heads, kv_heads, pos=RoPE(d_model // heads, max_seq_len, layout="half")
                ),
                mlp=SwiGLU(d_model),
                mixer_norm=RMSNorm(d_model),
                mlp_norm=RMSNorm(d_model),
            )
            for _ in range(n_layers)
        ]
        if scale_init:
            assert scale_residual_projections(blocks) == 2 * n_layers
        return Transformer(vocab, d_model, blocks, final_norm=RMSNorm(d_model))

    return _assemble
