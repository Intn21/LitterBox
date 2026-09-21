"""From a :class:`ModelConfig` to a model.

This is the one place a name in a YAML file becomes a module. Everything it
builds is handed to something else as an instance: a positional strategy to a
mixer, a mixer and an MLP and two norms to a block, the blocks to the backbone.
Nothing downstream constructs anything, which is what lets every layer of a
hybrid differ — mixer, KV heads, window, positional strategy — from config alone.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from litterbox.model.block import TransformerBlock, scale_residual_projections
from litterbox.model.mlp import SwiGLU
from litterbox.model.norm import RMSNorm
from litterbox.model.registry import available_mixers, get_mixer
from litterbox.model.transformer import Transformer
from litterbox.positional import Learned, NoPE, PositionalEncoding, RoPE, Sinusoidal
from litterbox.utils.config import LayerConfig, ModelConfig

# One implementation behind each today; ROADMAP step 5 adds a second to both.
MLPS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "swiglu": lambda cfg: SwiGLU(cfg.d_model, cfg.mlp.hidden_mult),
}
NORMS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "rmsnorm": lambda cfg: RMSNorm(cfg.d_model, cfg.norm.eps),
}


def resolve_mixer(name: str, tier: str) -> type:
    """The mixer class for ``name``, preferring its fused twin under ``tier="fast"``.

    A mixer with no fast twin yet is used as-is, so ``tier: fast`` on a hybrid
    speeds up the layers it can and leaves the rest alone.
    """
    if tier == "fast" and f"{name}_fast" in available_mixers():
        name = f"{name}_fast"
    return get_mixer(name)


def build_positional(name: str, head_dim: int, cfg: ModelConfig) -> PositionalEncoding:
    """A fresh per-layer strategy instance — one per block, as the config reads."""
    if name == "rope":
        return RoPE(head_dim, cfg.max_seq_len, layout=cfg.rope.layout, base=cfg.rope.base)
    if name == "nope":
        return NoPE()
    raise ValueError(f"unknown per-layer positional strategy {name!r}")


def build_block(layer: LayerConfig, cfg: ModelConfig, index: int = 0) -> TransformerBlock:
    """One block from one (already tiled) ``layer_pattern`` entry."""
    args = layer.mixer_args
    if layer.pos is not None:
        heads = args.get("heads")
        if heads is None:
            raise ValueError(f"layer {index} ({layer.mixer}): `pos` needs `heads` to size RoPE")
        args["pos"] = build_positional(layer.pos, cfg.d_model // heads, cfg)
    mixer_cls = resolve_mixer(layer.mixer, cfg.tier)
    try:
        mixer = mixer_cls(cfg.d_model, **args)
    except TypeError as e:
        # No mixer takes **kwargs, so a misspelled key lands here. Say which layer.
        raise TypeError(f"layer {index} ({layer.mixer}): {e}") from e
    return TransformerBlock(
        mixer=mixer,
        mlp=MLPS[cfg.mlp.type](cfg),
        mixer_norm=NORMS[cfg.norm.type](cfg),
        mlp_norm=NORMS[cfg.norm.type](cfg),
    )


def build_model(cfg: ModelConfig) -> Transformer:
    """Instantiate the model a config describes."""
    blocks = [build_block(layer, cfg, i) for i, layer in enumerate(cfg.layers)]
    if cfg.scale_residual_init:
        scale_residual_projections(blocks)

    pos: PositionalEncoding | None = None
    if cfg.pos is not None:
        if cfg.pos.type == "learned":
            pos = Learned(cfg.d_model, cfg.max_seq_len)
        else:
            pos = Sinusoidal(cfg.d_model, cfg.max_seq_len, learnable=cfg.pos.learnable)

    model = Transformer(
        cfg.vocab_size,
        cfg.d_model,
        blocks,
        pos=pos,
        final_norm=NORMS[cfg.norm.type](cfg),
        tie_embeddings=cfg.tie_embeddings,
    )
    model.max_seq_len = cfg.max_seq_len
    return model
