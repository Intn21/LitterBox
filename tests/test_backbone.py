"""Backbone shell: embedding, tied LM head, and the empty-stack properties.

These run with zero blocks on purpose — the embedding → norm → head path has
testable properties of its own, and pinning them down now means a wrong loss
after the first mixer lands can only have come from the mixer.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from litterbox.model.transformer import Transformer

VOCAB, D_MODEL = 512, 64


def test_forward_shape_and_dtype():
    model = Transformer(VOCAB, D_MODEL)
    ids = torch.randint(0, VOCAB, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, VOCAB)
    assert logits.dtype == torch.float32


def test_tied_head_is_the_same_parameter():
    tied = Transformer(VOCAB, D_MODEL, tie_embeddings=True)
    assert tied.lm_head.weight is tied.tok_emb.weight

    untied = Transformer(VOCAB, D_MODEL, tie_embeddings=False)
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.n_params - tied.n_params == VOCAB * D_MODEL


def test_initial_loss_is_ln_vocab():
    """The Step 1 exit check, on the blockless shell.

    Near-flat logits at init put the first loss at ln(vocab_size). A loss
    meaningfully below it means the targets are leaking into the inputs.
    """
    model = Transformer(VOCAB, D_MODEL)
    ids = torch.randint(0, VOCAB, (4, 128))
    targets = torch.randint(0, VOCAB, (4, 128))
    loss = F.cross_entropy(model(ids).flatten(0, 1), targets.flatten())
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.05)


@pytest.mark.parametrize("n_layers", [1, 4])
def test_initial_loss_is_ln_vocab_with_real_blocks(assemble, n_layers):
    """The Step 1 exit condition itself: a forward pass through real blocks —
    attention, SwiGLU, RMSNorm, RoPE — produces a loss, and it is ln(vocab).

    Real next-token targets, the inputs shifted by one, on purpose: that is the
    arrangement that would show a leak. If any path let position i see token
    i+1, this loss would start measurably below ln(vocab)."""
    torch.manual_seed(0)
    model = assemble(vocab=VOCAB, n_layers=n_layers)
    seq = torch.randint(0, VOCAB, (4, 65))
    logits = model(seq[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten())
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.05)


def test_every_parameter_in_an_assembled_model_gets_gradient(assemble):
    """No dead weights: one backward pass must reach every tensor, through
    every block. Tied embeddings count once."""
    model = assemble(vocab=VOCAB, n_layers=3)
    seq = torch.randint(0, VOCAB, (2, 33))
    F.cross_entropy(model(seq[:, :-1]).flatten(0, 1), seq[:, 1:].flatten()).backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert dead == []


def test_embedding_gradient_is_scatter_add():
    """Only rows of tokens present in the batch receive embedding gradient.

    Untied on purpose: with tied weights every row also participates in the
    LM head matmul and receives gradient through that path, so the pure
    gather/scatter-add property is only observable on a separate embedding.
    """
    model = Transformer(VOCAB, D_MODEL, tie_embeddings=False)
    ids = torch.tensor([[2, 0, 2]])
    targets = torch.tensor([[0, 2, 0]])
    loss = F.cross_entropy(model(ids).flatten(0, 1), targets.flatten())
    loss.backward()

    grad = model.tok_emb.weight.grad
    used = grad.abs().sum(dim=1)
    assert used[0] > 0 and used[2] > 0
    untouched = torch.ones(VOCAB, dtype=torch.bool)
    untouched[[0, 2]] = False
    assert torch.all(grad[untouched] == 0)


def test_tied_gradient_reaches_all_rows():
    """The flip side: tying routes head gradient into every embedding row."""
    model = Transformer(VOCAB, D_MODEL, tie_embeddings=True)
    ids = torch.tensor([[2, 0, 2]])
    targets = torch.tensor([[0, 2, 0]])
    loss = F.cross_entropy(model(ids).flatten(0, 1), targets.flatten())
    loss.backward()

    # Every row scored a candidate token in the softmax, so every row moves.
    assert torch.all(model.tok_emb.weight.grad.abs().sum(dim=1) > 0)
