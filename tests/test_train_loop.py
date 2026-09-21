"""The training loop: schedule, decay groups, exact resume, and the guards.

Everything runs on the CPU in fp32 against a tiny byte-level corpus, so the
numbers are deterministic and the exact-resume test can demand bit equality.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from litterbox.data import DataConfig, PackedDataset, prepare
from litterbox.infer import generate_uncached
from litterbox.model import dense_transformer
from litterbox.train import (
    RunConfig,
    evaluate,
    load_checkpoint,
    load_run_config,
    lr_at,
    param_groups,
    pick_device,
    save_checkpoint,
    train,
)
from litterbox.train.loop import ScheduleConfig

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "training"


@pytest.fixture
def corpus(tmp_path):
    """A small, learnable corpus: a repeating pattern a tiny model can pick up."""
    docs = [
        f"the {a} sat on the {b}. the {a} saw the {b}. " * 4
        for a in ("cat", "dog", "bird", "frog", "bear", "fox")
        for b in ("mat", "log", "hill", "rock", "bed")
    ]
    cfg = DataConfig(
        out_dir=str(tmp_path / "packed"),
        tokenizer={"type": "byte"},
        splits={"train": {"type": "text", "text": docs}, "val": {"type": "text", "text": docs[:6]}},
    )
    prepare(cfg, progress=False)
    return cfg.out_dir


def run_config(tmp_path, **training):
    base = {
        "seed": 0,
        "max_steps": 20,
        "batch_size": 4,
        "seq_len": 32,
        "optimizer": {"lr": 3e-3},
        "schedule": {"warmup_steps": 4},
        "precision": {"dtype": "float32"},
        "device": "cpu",
    }
    base.update(training)
    return RunConfig.model_validate(
        {
            "training": base,
            "data": {"config": "unused"},
            "logging": {
                "out_dir": str(tmp_path / "run"),
                "log_every": 5,
                "eval_every": 10,
                "eval_batches": 2,
                "checkpoint_every": 10,
            },
        }
    )


def tiny_model():
    torch.manual_seed(0)
    return dense_transformer(256, 32, 2, 4, 2, max_seq_len=32)


def datasets(path, seq_len=32):
    return PackedDataset(path, seq_len, split="train"), PackedDataset(path, seq_len, split="val")


# ----------------------------------------------------------------- schedule


def test_lr_schedule_warms_up_peaks_and_lands_on_the_floor():
    sched = ScheduleConfig(warmup_steps=10, min_lr_ratio=0.1)
    lrs = [lr_at(s, 100, 1.0, sched) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1)  # step 0 already moves: (0 + 1) / 10
    assert lrs[9] == pytest.approx(1.0)  # the last warmup step reaches the peak
    assert all(b > a for a, b in zip(lrs[:9], lrs[1:10], strict=True))  # rising through warmup
    assert all(b <= a for a, b in zip(lrs[10:], lrs[11:], strict=False))  # then only down
    assert lr_at(100, 100, 1.0, sched) == pytest.approx(0.1)  # the floor, not zero
    assert lr_at(5000, 100, 1.0, sched) == pytest.approx(0.1)  # and it stays there
    assert lr_at(55, 100, 1.0, sched) == pytest.approx(0.55)  # cosine midpoint: halfway down

    constant = ScheduleConfig(type="constant", warmup_steps=10)
    assert lr_at(50, 100, 1.0, constant) == 1.0


# ------------------------------------------------------------- decay groups


def test_param_groups_decay_matrices_and_spare_gains():
    model = tiny_model()
    decay, no_decay = param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0

    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"])
    gains = [p for n, p in model.named_parameters() if n.endswith("gain")]
    assert gains and all(any(g is p for p in no_decay["params"]) for g in gains)

    # Every parameter lands in exactly one group, and the tied matrix once.
    grouped = decay["params"] + no_decay["params"]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(p) for p in grouped}) == len(grouped)
    assert sum(p is model.tok_emb.weight for p in grouped) == 1


# ---------------------------------------------------------------- the loop


def test_training_lowers_the_loss_and_writes_a_log(tmp_path, corpus):
    train_ds, val_ds = datasets(corpus)
    cfg = run_config(tmp_path, max_steps=60)
    before = evaluate(tiny_model(), val_ds, batch_size=4, n_batches=2, device=torch.device("cpu"))

    result = train(tiny_model(), cfg, train_data=train_ds, val_data=val_ds, quiet=True)

    assert before == pytest.approx(math.log(256), abs=0.1)  # untrained: ln(vocab)
    assert result.val_loss < before - 1.0
    assert result.step == 60 and result.device == "cpu"

    lines = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    train_lines = [entry for entry in lines if "loss" in entry]
    assert {"step", "loss", "lr", "grad_norm", "tokens_per_sec"} <= set(train_lines[0])
    assert [entry["step"] for entry in lines if "val_loss" in entry] == [10, 20, 30, 40, 50, 60]
    assert (tmp_path / "run" / "latest.pt").exists()


def test_a_resumed_run_is_bit_identical_to_an_uninterrupted_one(tmp_path, corpus):
    """Batches are a function of (seed, step) and nothing in the model is
    stochastic, so stopping at step 10 and resuming must reproduce the
    uninterrupted run exactly — weights, optimizer state and all."""
    train_ds, val_ds = datasets(corpus)

    straight = tiny_model()
    train(straight, run_config(tmp_path / "a"), train_data=train_ds, val_data=val_ds, quiet=True)

    cfg = run_config(tmp_path / "b")
    first = tiny_model()
    stopped = train(first, cfg, train_data=train_ds, val_data=val_ds, stop_at_step=10, quiet=True)
    assert stopped.step == 10

    resumed = tiny_model()  # a fresh process: new model object, state comes from disk
    finished = train(resumed, cfg, train_data=train_ds, val_data=val_ds, quiet=True)
    assert finished.step == 20

    for (name, a), (_, b) in zip(
        straight.state_dict().items(), resumed.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_gradient_accumulation_runs_the_same_number_of_optimizer_steps(tmp_path, corpus):
    train_ds, val_ds = datasets(corpus)
    cfg = run_config(tmp_path, batch_size=2, grad_accum_steps=3, max_steps=10)
    result = train(tiny_model(), cfg, train_data=train_ds, val_data=val_ds, quiet=True)
    assert result.step == 10 and math.isfinite(result.train_loss)


def test_evaluate_is_repeatable_and_restores_training_mode(corpus):
    _, val_ds = datasets(corpus)
    model = tiny_model().train()
    device = torch.device("cpu")
    a = evaluate(model, val_ds, batch_size=4, n_batches=3, device=device)
    b = evaluate(model, val_ds, batch_size=4, n_batches=3, device=device)
    assert a == b  # the same fixed batches every time
    assert model.training


# ------------------------------------------------------------------ guards


def test_a_dead_gradient_stops_the_run(tmp_path, corpus):
    """A gradient norm of exactly zero never shows in the loss — it just stops
    moving. Seen on an oversubscribed Apple GPU; the loop must make it loud."""

    class Dead(nn.Module):
        vocab_size = 256

        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(256))

        def forward(self, ids):
            return (self.w * 0.0).expand(*ids.shape, 256)

    train_ds, _ = datasets(corpus)
    with pytest.raises(FloatingPointError, match="gradient norm is 0.0"):
        train(Dead(), run_config(tmp_path), train_data=train_ds, quiet=True)


def test_a_vocabulary_larger_than_the_embedding_is_refused(tmp_path, corpus):
    train_ds, _ = datasets(corpus)
    too_small = dense_transformer(100, 32, 1, 4, 2, max_seq_len=32)
    with pytest.raises(ValueError, match="embedding has 100 rows"):
        train(too_small, run_config(tmp_path), train_data=train_ds, quiet=True)


def test_checkpoint_round_trip_is_atomic_and_complete(tmp_path):
    cfg = run_config(tmp_path)
    model = tiny_model()
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-3)
    model(torch.randint(0, 256, (2, 8))).sum().backward()
    opt.step()

    path = tmp_path / "ckpt" / "latest.pt"
    save_checkpoint(path, model, opt, step=7, cfg=cfg)
    assert path.exists() and not path.with_suffix(".tmp").exists()

    other = dense_transformer(256, 32, 2, 4, 2, max_seq_len=32)  # different init
    other_opt = torch.optim.AdamW(param_groups(other, 0.1), lr=1e-3)
    assert load_checkpoint(path, other, other_opt) == 7
    for (name, a), (_, b) in zip(
        model.state_dict().items(), other.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert other_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


# ------------------------------------------------------------------ configs


def test_shipped_training_configs_validate_and_take_overrides():
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.stem in {"base", "tinystories-small"}:
            cfg = load_run_config(path)
            assert Path(cfg.data.config).name.endswith(".yaml")

    cfg = load_run_config(
        CONFIG_DIR / "tinystories-small.yaml", ["training.max_steps=7", "logging.out_dir=runs/x"]
    )
    assert cfg.training.max_steps == 7 and cfg.logging.out_dir == "runs/x"
    with pytest.raises(ValidationError, match="max_step"):
        load_run_config(CONFIG_DIR / "tinystories-small.yaml", ["training.max_step=7"])
    with pytest.raises(ValidationError, match="float16"):
        load_run_config(CONFIG_DIR / "tinystories-small.yaml", ["training.precision.dtype=float16"])


def test_pick_device_honours_an_explicit_choice():
    assert pick_device("cpu") == torch.device("cpu")
    assert pick_device("auto").type in {"cuda", "mps", "cpu"}


# --------------------------------------------------------------- generation


def test_generate_uncached_greedy_is_deterministic_and_matches_top_k_one():
    model = tiny_model()
    prompt = torch.randint(0, 256, (2, 5))
    a = generate_uncached(model, prompt, 10, max_context=32, temperature=0)
    b = generate_uncached(model, prompt, 10, max_context=32, temperature=0)
    one = generate_uncached(model, prompt, 10, max_context=32, temperature=1.0, top_k=1)
    assert a.shape == (2, 15) and torch.equal(a[:, :5], prompt)
    assert torch.equal(a, b) and torch.equal(a, one)
    assert model.training  # restored


def test_generate_uncached_sampling_is_seeded_and_slides_past_the_context_limit():
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 4))

    def sample(seed):
        gen = torch.Generator().manual_seed(seed)
        return generate_uncached(model, prompt, 12, max_context=32, top_k=20, generator=gen)

    assert torch.equal(sample(1), sample(1))
    assert not torch.equal(sample(1), sample(2))

    # 4 + 40 tokens is past RoPE's 32 positions; the window must slide, not raise.
    long = generate_uncached(model, prompt, 40, max_context=32, temperature=0)
    assert long.shape == (1, 44)


def test_generate_uncached_stops_at_eos():
    model = tiny_model()
    prompt = torch.randint(0, 256, (1, 4))
    first = generate_uncached(model, prompt, 1, max_context=32, temperature=0)[0, -1].item()
    out = generate_uncached(model, prompt, 20, max_context=32, temperature=0, eos_id=first)
    assert out.shape == (1, 5)  # stopped right after producing it
