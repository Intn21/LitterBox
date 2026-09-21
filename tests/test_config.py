"""Model configs: the schema, layer-pattern tiling, and the builder.

This is where ROADMAP step 4's claim gets checked structurally: a hybrid is a
short list in a YAML file, and the code that builds, runs and generates from it
is the code that handles a plain dense model.
"""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from litterbox.infer import KVCache, RollingKVCache, generate, generate_uncached
from litterbox.model import build_model, dense_transformer
from litterbox.model.mixers.fast.full_attention import FastFullAttention
from litterbox.model.mixers.fast.sliding_window import FastSlidingWindowAttention
from litterbox.model.mixers.reference.full_attention import FullAttention
from litterbox.model.mixers.reference.sliding_window import SlidingWindowAttention
from litterbox.positional import Learned, NoPE, RoPE
from litterbox.utils.config import ModelConfig, load_model_config

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "models"
BUILDABLE = ["tiny-dense-100m", "tiny-swa-100m", "tiny-swa-hybrid-3to1"]
NOT_YET = ["tiny-gdn-hybrid-3to1", "tiny-dsa-100m"]
TRIO = ["tinystories-dense", "tinystories-swa", "tinystories-swa-hybrid"]

SWA = {"mixer": "sliding_window", "heads": 4, "kv_heads": 2, "window": 8, "pos": "rope"}
FULL = {"mixer": "full_attention", "heads": 4, "kv_heads": 1, "pos": "nope"}


def small(**kw):
    base = dict(
        d_model=32, n_layers=4, vocab_size=64, max_seq_len=48, layer_pattern=[SWA, SWA, SWA, FULL]
    )
    base.update(kw)
    return ModelConfig(**base)


# ------------------------------------------------------------- shipped configs


def test_every_shipped_model_config_loads_under_the_strict_schema():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {p.stem for p in paths} >= set(BUILDABLE + NOT_YET + TRIO)
    for path in paths:
        cfg = load_model_config(path)
        assert len(cfg.layers) == cfg.n_layers, path.name


@pytest.mark.parametrize("name", BUILDABLE)
def test_the_100m_configs_build(name):
    """On the meta device: shapes and wiring, no memory."""
    cfg = load_model_config(CONFIG_DIR / f"{name}.yaml")
    with torch.device("meta"):
        model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 115e6 < n_params < 130e6
    assert len(model.blocks) == 12 and model.max_seq_len == cfg.max_seq_len


def test_a_window_adds_no_parameters_and_fewer_kv_heads_removes_some():
    def count(name):
        with torch.device("meta"):
            model = build_model(load_model_config(CONFIG_DIR / f"{name}.yaml"))
        return sum(p.numel() for p in model.parameters())

    assert count("tiny-swa-100m") == count("tiny-dense-100m")
    assert count("tiny-swa-hybrid-3to1") < count(
        "tiny-dense-100m"
    )  # its full layers use 2 KV heads


@pytest.mark.parametrize("name", NOT_YET)
def test_configs_for_unwritten_mixers_load_but_do_not_build(name):
    cfg = load_model_config(CONFIG_DIR / f"{name}.yaml")
    with pytest.raises(NotImplementedError):
        build_model(cfg)


# --------------------------------------------------------------------- tiling


def test_layer_pattern_tiles_to_n_layers_in_order():
    model = build_model(small(n_layers=8))
    kinds = [type(b.mixer) for b in model.blocks]
    assert kinds == [SlidingWindowAttention] * 3 + [FullAttention] + [
        SlidingWindowAttention
    ] * 3 + [FullAttention]


def test_every_layer_gets_its_own_settings_and_its_own_instances():
    model = build_model(small())
    windowed, full = model.blocks[0].mixer, model.blocks[3].mixer
    assert (windowed.window, windowed.kv_heads, full.kv_heads) == (8, 2, 1)
    assert isinstance(windowed.pos, RoPE) and windowed.pos.head_dim == 8
    assert isinstance(full.pos, NoPE)  # `pos: nope` on the full layer, as Kimi Linear does
    assert model.blocks[0].mixer.pos is not model.blocks[1].mixer.pos
    assert model.blocks[0].mixer_norm is not model.blocks[0].mlp_norm


def test_a_pattern_that_does_not_tile_is_refused():
    with pytest.raises(ValidationError, match="does not tile into n_layers=6"):
        small(n_layers=6)
    assert len(small(n_layers=12).layers) == 12


# ----------------------------------------------------------------- strictness


def test_schema_refuses_what_it_cannot_mean():
    with pytest.raises(ValidationError, match="d_modle"):
        ModelConfig(d_modle=32, n_layers=1, vocab_size=8, max_seq_len=8, layer_pattern=[FULL])
    with pytest.raises(ValidationError, match="unknown mixer 'full_attn'; available:"):
        small(layer_pattern=[{**FULL, "mixer": "full_attn"}], n_layers=1)
    with pytest.raises(ValidationError, match="not divisible by heads=5"):
        small(layer_pattern=[{**FULL, "heads": 5}], n_layers=1)
    with pytest.raises(ValidationError, match="gelu"):
        small(mlp={"type": "gelu"})


def test_an_additive_strategy_named_per_layer_points_at_the_right_place():
    with pytest.raises(ValidationError, match="set for the whole model as `pos: {type: learned}`"):
        small(layer_pattern=[{**FULL, "pos": "learned"}], n_layers=1)


def test_a_misspelled_mixer_argument_fails_at_build_naming_the_layer():
    """The schema cannot know every mixer's arguments, so extras pass through —
    but no mixer takes **kwargs, so a typo still cannot be swallowed."""
    cfg = small(layer_pattern=[FULL, {**SWA, "windw": 8}], n_layers=2)
    with pytest.raises(TypeError, match=r"layer 1 \(sliding_window\)"):
        build_model(cfg)


# ------------------------------------------------------------- builder options


def test_model_level_additive_positions():
    model = build_model(small(pos={"type": "learned"}))
    assert isinstance(model.pos, Learned) and model.pos.table.shape == (48, 32)
    assert build_model(small()).pos is None


def test_tier_fast_swaps_in_the_fused_twins_with_identical_numbers():
    torch.manual_seed(0)
    reference = build_model(small())
    fast = build_model(small(tier="fast"))
    assert isinstance(fast.blocks[0].mixer, FastSlidingWindowAttention)
    assert isinstance(fast.blocks[3].mixer, FastFullAttention)
    fast.load_state_dict(reference.state_dict())  # same keys: checkpoints move between tiers
    ids = torch.randint(0, 64, (2, 24))
    assert torch.allclose(fast(ids), reference(ids), atol=1e-4)


def test_dense_transformer_is_the_same_model_the_yaml_would_make():
    torch.manual_seed(0)
    by_hand = dense_transformer(64, 32, 2, 4, 2, max_seq_len=48)
    torch.manual_seed(0)
    by_config = build_model(
        small(
            n_layers=2,
            layer_pattern=[{"mixer": "full_attention", "heads": 4, "kv_heads": 2, "pos": "rope"}],
        )
    )
    a, b = by_hand.state_dict(), by_config.state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------- the hybrid, end to end


def test_a_hybrid_generates_from_mixed_caches_exactly_what_it_generates_without_them():
    """Generation never learns that layer 3 differs from layer 0: it asks each
    block for its state and gets a rolling cache from some and a full one from
    others."""
    torch.manual_seed(0)
    model = build_model(small(layer_pattern=[SWA, SWA, SWA, {**FULL, "pos": "rope"}]))
    states = model.init_states(1, 40)
    assert [type(s.kv) for s in states] == [RollingKVCache] * 3 + [KVCache]

    prompt = torch.randint(0, 64, (2, 6))
    cached = generate(model, prompt, 34, max_context=48, temperature=0)  # well past the window of 8
    assert torch.equal(cached, generate_uncached(model, prompt, 34, max_context=48, temperature=0))


def test_in_a_hybrid_only_the_full_layers_keep_growing():
    model = build_model(small(layer_pattern=[SWA, SWA, SWA, {**FULL, "pos": "rope"}])).eval()
    states = model.init_states(1, 40)
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 40)), states)
    assert [s.kv.length for s in states] == [8, 8, 8, 40]
    per_layer = [b.mixer.state_bytes(40) for b in model.blocks]
    assert per_layer[0] == per_layer[1] == per_layer[2] < per_layer[3]
