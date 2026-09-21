"""Model config: loading and schema validation.

Validation is strict on purpose: an unknown key or a ``layer_pattern`` that does
not tile cleanly into ``n_layers`` should fail at load time, not surface as a
confusing shape error thirty seconds into a run.

The schema describes *what* a model is. Turning it into modules is
:mod:`litterbox.model.build`'s job, and the split is deliberate: a config can be
loaded, validated, diffed and hashed without importing torch modules or
allocating a single parameter.

One part is deliberately *not* strict here. A ``layer_pattern`` entry names a
mixer and then carries whatever arguments that mixer takes — ``window`` for
sliding-window attention, an ``indexer`` block for DSA — which this file cannot
know without importing every mixer. Those extra keys are passed to the mixer's
constructor, and since no mixer accepts ``**kwargs``, a misspelled one still
fails, at build time, naming the layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Positional strategies that act inside attention, and so are chosen per layer.
PER_LAYER_POSITIONAL = ("rope", "nope")
# ...and those that are added to the residual stream once, at the input.
ADDITIVE_POSITIONAL = ("learned", "sinusoidal")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LayerConfig(BaseModel):
    """One ``layer_pattern`` entry: a mixer, a positional choice, and the mixer's own arguments."""

    model_config = ConfigDict(extra="allow")

    mixer: str
    pos: str | None = None  # omitted for mixers that take no positional strategy

    @field_validator("mixer")
    @classmethod
    def _mixer_is_registered(cls, name: str) -> str:
        # Imported here, not at module level: litterbox.model imports this module.
        from litterbox.model.registry import available_mixers

        if name not in available_mixers():
            raise ValueError(f"unknown mixer {name!r}; available: {', '.join(available_mixers())}")
        return name

    @field_validator("pos")
    @classmethod
    def _pos_is_per_layer(cls, name: str | None) -> str | None:
        if name in ADDITIVE_POSITIONAL:
            raise ValueError(
                f"{name!r} is an additive strategy: it is added to the residual stream once, at "
                f"the input, so it is set for the whole model as `pos: {{type: {name}}}`, not per "
                f"layer. Per-layer choices are {', '.join(PER_LAYER_POSITIONAL)}."
            )
        if name is not None and name not in PER_LAYER_POSITIONAL:
            raise ValueError(
                f"unknown per-layer positional strategy {name!r}; "
                f"choose from {', '.join(PER_LAYER_POSITIONAL)}"
            )
        return name

    @property
    def mixer_args(self) -> dict[str, Any]:
        """Everything in the entry that is neither ``mixer`` nor ``pos``."""
        return dict(self.model_extra or {})


class MLPConfig(_Strict):
    type: Literal["swiglu"] = "swiglu"
    hidden_mult: float = Field(default=4.0, gt=0)


class NormConfig(_Strict):
    type: Literal["rmsnorm"] = "rmsnorm"
    eps: float = Field(default=1e-5, gt=0)


class RopeConfig(_Strict):
    # No default for layout at the class that does the rotating, on purpose. It
    # gets one here because a from-scratch model has no checkpoint to disagree
    # with; loading someone else's weights means setting it to match theirs.
    layout: Literal["half", "interleaved"] = "half"
    base: float = Field(default=10000.0, gt=0)


class AdditivePositionalConfig(_Strict):
    type: Literal["learned", "sinusoidal"]
    learnable: bool = False  # sinusoidal only: start from the sinusoid, then train it


class ModelConfig(_Strict):
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    layer_pattern: list[LayerConfig] = Field(min_length=1)
    mlp: MLPConfig = MLPConfig()
    norm: NormConfig = NormConfig()
    rope: RopeConfig = RopeConfig()
    pos: AdditivePositionalConfig | None = None
    tie_embeddings: bool = True
    scale_residual_init: bool = True
    # "fast" swaps each mixer for its fused twin where one exists. Same
    # parameters, same numbers, different speed and memory; checkpoints move
    # between the two.
    tier: Literal["reference", "fast"] = "reference"

    @model_validator(mode="after")
    def _pattern_tiles(self) -> ModelConfig:
        n = len(self.layer_pattern)
        if self.n_layers % n != 0:
            raise ValueError(
                f"layer_pattern has {n} entries, which does not tile into n_layers="
                f"{self.n_layers}. A hybrid's ratio should hold for the whole stack, not be "
                f"cut off partway: use a multiple of {n}."
            )
        for i, layer in enumerate(self.layer_pattern):
            heads = layer.mixer_args.get("heads")
            if heads is not None and self.d_model % heads != 0:
                raise ValueError(
                    f"layer_pattern[{i}] ({layer.mixer}): d_model={self.d_model} is not "
                    f"divisible by heads={heads}"
                )
        return self

    @property
    def layers(self) -> list[LayerConfig]:
        """``layer_pattern`` repeated to ``n_layers`` — one entry per block, in order."""
        return self.layer_pattern * (self.n_layers // len(self.layer_pattern))


def load_model_config(path: str | Path, overrides: list[str] | None = None) -> ModelConfig:
    """Read and validate ``configs/models/*.yaml``.

    ``overrides`` are OmegaConf dotlist entries applied under the ``model`` key:
    ``["d_model=128", "n_layers=4"]``.
    """
    cfg = OmegaConf.load(path)
    if "model" not in cfg:
        raise ValueError(f"{path}: expected a top-level `model:` block")
    model = cfg.model
    if overrides:
        model = OmegaConf.merge(model, OmegaConf.from_dotlist(list(overrides)))
    return ModelConfig.model_validate(OmegaConf.to_container(model, resolve=True))


load_config = load_model_config  # the name the original scaffolding used
