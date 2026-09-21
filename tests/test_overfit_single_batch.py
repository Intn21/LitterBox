"""Single-batch overfit: every config can memorize one batch.

The cheapest end-to-end signal that a model config is trainable at all. It
catches dead gradients, wrong-way masks, and misconfigured norms in seconds
rather than after a night of training."""

import pathlib

import pytest
import torch
import torch.nn.functional as F

stub = pytest.mark.xfail(
    reason="Scaffolding: needs utils/config.py to build a model from YAML.",
    strict=False,
)

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs" / "models"


def test_a_hand_assembled_model_overfits_one_batch(assemble):
    """The config-free version: build the dense model by hand, show it one
    fixed batch of random tokens over and over, and require it to memorise
    them. Random tokens have no pattern to generalise from, so the only way
    down is to use attention to look up what came before — which exercises
    every piece: embeddings, RoPE, attention, SwiGLU, the norms, the head."""
    torch.manual_seed(0)
    vocab = 64
    model = assemble(vocab=vocab, d_model=64, heads=4, kv_heads=2, n_layers=2)
    seq = torch.randint(0, vocab, (4, 33))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for _ in range(300):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert first > 3.5  # started near ln(64) = 4.16, not already memorised
    assert loss.item() < 0.05


@stub
@pytest.mark.slow
def test_every_model_config_overfits_one_batch():
    """Loss must fall near zero on a single fixed batch, for each config."""
    raise NotImplementedError("Milestone 1")
