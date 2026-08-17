"""Single-batch overfit: every config can memorize one batch.

The cheapest end-to-end signal that a model config is trainable at all. It
catches dead gradients, wrong-way masks, and misconfigured norms in seconds
rather than after a night of training."""

import pathlib

import pytest

pytestmark = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)

CONFIG_DIR = pathlib.Path(__file__).parent.parent / "configs" / "models"


@pytest.mark.slow
def test_every_model_config_overfits_one_batch():
    """Loss must fall near zero on a single fixed batch, for each config."""
    raise NotImplementedError("Milestone 1")
