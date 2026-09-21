"""Single-batch overfit: every config can memorize one batch.

The cheapest end-to-end signal that a model config is trainable at all. It
catches dead gradients, wrong-way masks, and misconfigured norms in seconds
rather than after a night of training."""

import pathlib

import pytest
import torch
import torch.nn.functional as F

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


TRIO = ["tinystories-dense", "tinystories-swa", "tinystories-swa-hybrid"]
# Shrunk so the test takes a second, but the layer_pattern — the only thing that
# differs between the three files — is used exactly as written.
SHRINK = ["d_model=64", "vocab_size=64", "max_seq_len=64"]


@pytest.mark.parametrize("name", TRIO)
def test_every_model_config_overfits_one_batch(name):
    """ROADMAP step 4's condition, in miniature: three configs, one code path.

    Everything below this line is identical for all three. The YAML chooses the
    mixers; building, the forward pass, the loss and the optimizer never learn
    which ones they got."""
    from litterbox.model import build_model
    from litterbox.utils.config import load_model_config

    torch.manual_seed(0)
    model = build_model(load_model_config(CONFIG_DIR / f"{name}.yaml", SHRINK))
    seq = torch.randint(0, 64, (4, 49))
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for _ in range(300):
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert first > 3.5
    assert loss.item() < 0.1
