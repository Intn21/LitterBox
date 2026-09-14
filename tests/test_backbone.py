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
