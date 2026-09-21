"""Parallel/recurrent duality: training-mode and decode-mode must agree.

The full-sequence parallel (or chunked) forward and the step-by-step
recurrence with MixerState must produce the same outputs. This is the single
most bug-prone property of the DeltaNet family and the cheapest test to keep
green from day one.

Tolerance rather than bit equality: a one-token step and a full-sequence pass
multiply different-shaped matrices, and floating-point sums are not associative.
The *tokens* a model generates are still required to match exactly — see
test_generation.py."""

import pytest
import torch

from litterbox.model import get_mixer
from litterbox.positional import RoPE

stub = pytest.mark.xfail(
    reason="Scaffolding: mixers are stubs. Remove this marker as they land.",
    strict=False,
)

D_MODEL, HEADS, KV_HEADS, SEQ, BATCH, WINDOW = 32, 4, 2, 20, 2, 6
HEAD_DIM = D_MODEL // HEADS


def build(name):
    """One small instance of each mixer that has landed. Add a line per mixer."""
    torch.manual_seed(0)
    pos = RoPE(HEAD_DIM, 64, layout="half")
    if name in ("full_attention", "full_attention_fast"):
        return get_mixer(name)(D_MODEL, HEADS, KV_HEADS, pos=pos)
    if name in ("sliding_window", "sliding_window_fast"):
        # A window well inside SEQ, so every test crosses the point where the
        # cache starts forgetting.
        return get_mixer(name)(D_MODEL, HEADS, KV_HEADS, pos=pos, window=WINDOW)
    raise KeyError(name)


LANDED = ["full_attention", "full_attention_fast", "sliding_window", "sliding_window_fast"]


def run_in_chunks(mixer, x, chunks):
    """Feed ``x`` through ``mixer`` in the given chunk sizes, carrying state."""
    state = mixer.init_state(x.shape[0], x.shape[1])
    outs, pos = [], 0
    for n in chunks:
        out, state = mixer(x[:, pos : pos + n], state, pos_offset=pos)
        outs.append(out)
        pos += n
    assert pos == x.shape[1]
    return torch.cat(outs, dim=1), state


@pytest.mark.parametrize("name", LANDED)
def test_prefill_equals_sequential_decode(name):
    """One parallel pass over N tokens == N single-token steps with state."""
    mixer = build(name)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    parallel, none_state = mixer(x)
    stepwise, state = run_in_chunks(mixer, x, [1] * SEQ)
    assert none_state is None
    assert torch.allclose(stepwise, parallel, atol=1e-5)
    assert state.kv.total == SEQ


@pytest.mark.parametrize("name", LANDED)
@pytest.mark.parametrize("chunks", [[SEQ], [12, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 1, 5, 1]])
def test_partial_prefill_then_decode(name, chunks):
    """Prefill k tokens, decode the rest; must equal a full parallel pass.

    The third chunking appends a multi-token chunk to a non-empty cache, which
    is the one case where the mask is a *shifted* triangle — neither the square
    of training nor the single row of decoding."""
    mixer = build(name)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    parallel, _ = mixer(x)
    chunked, _ = run_in_chunks(mixer, x, chunks)
    assert torch.allclose(chunked, parallel, atol=1e-5)


@stub
def test_chunked_forward_matches_recurrent():
    """Chunked linear-attention forward vs the pure recurrence."""
    raise NotImplementedError("Milestone 3")


@pytest.mark.parametrize("name", LANDED)
def test_state_bytes_per_token_is_honest(name):
    """Reported bytes/token must match measured state growth over context."""
    mixer = build(name)
    x = torch.randn(1, SEQ, D_MODEL)
    state = mixer.init_state(1, SEQ)
    used = []
    for t in range(SEQ):
        _, state = mixer(x[:, t : t + 1], state, pos_offset=t)
        used.append(state.kv.nbytes_used)
    # What the mixer claims its state costs after n tokens is what it measurably costs.
    assert used == [mixer.state_bytes(n) for n in range(1, SEQ + 1)]
    assert used[0] == mixer.state_bytes_per_token
    # ...and it is the KV heads that set the per-token cost, not the query heads.
    assert mixer.state_bytes_per_token == 2 * KV_HEADS * HEAD_DIM * 4

    growth = [b - a for a, b in zip(used, used[1:], strict=False)]
    if "sliding_window" in name:
        # Grows like full attention until the window is full, then not at all.
        assert growth[: WINDOW - 1] == [mixer.state_bytes_per_token] * (WINDOW - 1)
        assert set(growth[WINDOW - 1 :]) == {0}
    else:
        assert set(growth) == {mixer.state_bytes_per_token}  # forever


@pytest.mark.parametrize("name", LANDED)
def test_decoding_with_the_wrong_position_is_refused(name):
    """The quiet bug: a lone token decoded at the default pos_offset=0 is turned
    as if it opened the document. For a full-attention cache the position *is*
    the cache length, so the mixer can catch it."""
    mixer = build(name)
    x = torch.randn(1, 6, D_MODEL)
    state = mixer.init_state(1, 16)
    _, state = mixer(x[:, :5], state, pos_offset=0)
    with pytest.raises(ValueError, match="pos_offset=0 but the cache has already seen 5"):
        mixer(x[:, 5:], state)
    out, _ = mixer(x[:, 5:], state, pos_offset=5)
    assert torch.allclose(out, mixer(x)[0][:, 5:], atol=1e-5)
