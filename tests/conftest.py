"""Shared fixtures."""

import pytest

from litterbox.model import dense_transformer


@pytest.fixture
def assemble():
    """Build a small dense model by hand — no config system involved.

    A thin wrapper over :func:`litterbox.model.dense_transformer` with test-sized
    defaults, so every model-level test shares one construction with the
    training loop.
    """

    def _assemble(
        vocab=256, d_model=64, heads=4, kv_heads=2, n_layers=2, max_seq_len=128, scale_init=True
    ):
        return dense_transformer(
            vocab,
            d_model,
            n_layers,
            heads,
            kv_heads,
            max_seq_len=max_seq_len,
            scale_init=scale_init,
        )

    return _assemble
