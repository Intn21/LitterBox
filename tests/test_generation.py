"""ROADMAP step 3's exit condition: cached generation matches cache-free
generation exactly, and nothing a token does can reach backwards."""

import pytest
import torch

from litterbox.infer import generate, generate_uncached
from litterbox.model import dense_transformer
from litterbox.positional import Learned, Sinusoidal

VOCAB, CONTEXT = 200, 96
TIERS = ["full_attention", "full_attention_fast"]


def model(mixer="full_attention", kv_heads=2, **kw):
    torch.manual_seed(0)
    return dense_transformer(VOCAB, 64, 3, 4, kv_heads, max_seq_len=CONTEXT, mixer=mixer, **kw)


@pytest.mark.parametrize("mixer", TIERS)
@pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
def test_cached_generation_matches_uncached_exactly(mixer, kv_heads):
    m = model(mixer, kv_heads)
    prompt = torch.randint(0, VOCAB, (3, 9))

    greedy = generate(m, prompt, 50, max_context=CONTEXT, temperature=0)
    oracle = generate_uncached(m, prompt, 50, max_context=CONTEXT, temperature=0)
    assert torch.equal(greedy, oracle)

    def sampled(fn):
        gen = torch.Generator().manual_seed(11)
        return fn(m, prompt, 50, max_context=CONTEXT, temperature=0.9, top_k=25, generator=gen)

    assert torch.equal(sampled(generate), sampled(generate_uncached))
    assert m.training  # both restore the mode they found


@pytest.mark.parametrize(
    "make_pos", [lambda: Learned(64, CONTEXT), lambda: Sinusoidal(64, CONTEXT)]
)
def test_additive_positions_follow_pos_offset_through_the_backbone(make_pos):
    """A learned or sinusoidal table acts in the backbone, not in attention. A
    decoded token must be given the row for its real position, not row 0."""
    m = model()
    m.pos = make_pos()
    prompt = torch.randint(0, VOCAB, (2, 6))
    assert torch.equal(
        generate(m, prompt, 30, max_context=CONTEXT, temperature=0),
        generate_uncached(m, prompt, 30, max_context=CONTEXT, temperature=0),
    )


def test_logits_from_a_stateful_pass_match_the_training_pass():
    m = model().eval()
    ids = torch.randint(0, VOCAB, (2, 24))
    reference = m(ids)

    states = m.init_states(2, 24)
    prefill = m(ids[:, :10], states, pos_offset=0)
    rest = [m(ids[:, t : t + 1], states, pos_offset=t) for t in range(10, 24)]
    stateful = torch.cat([prefill, *rest], dim=1)

    assert torch.allclose(stateful, reference, atol=1e-4)
    assert [s.kv.length for s in states] == [24, 24, 24]  # one cache per block, all in step
    assert states[0].kv is not states[1].kv


def test_nothing_generated_later_can_change_what_came_before():
    """Perturb the token at position t of a prompt: every logit before t is
    bit-identical, because the cache for those positions was filled from
    exactly the same inputs."""
    m = model().eval()
    ids = torch.randint(0, VOCAB, (1, 16))
    poked = ids.clone()
    poked[0, 11] = (poked[0, 11] + 1) % VOCAB

    def stepwise(seq):
        states = m.init_states(1, 16)
        return torch.cat([m(seq[:, t : t + 1], states, pos_offset=t) for t in range(16)], dim=1)

    before, after = stepwise(ids), stepwise(poked)
    assert torch.equal(before[:, :11], after[:, :11])
    assert not torch.allclose(before[:, 11:], after[:, 11:])


def test_generate_refuses_what_a_cache_cannot_do():
    m = model()
    prompt = torch.randint(0, VOCAB, (1, 10))
    with pytest.raises(ValueError, match="cannot slide"):
        generate(m, prompt, CONTEXT, max_context=CONTEXT, temperature=0)
    # the uncached oracle slides its window and carries on
    assert generate_uncached(m, prompt, CONTEXT, max_context=CONTEXT, temperature=0).shape[1] == (
        10 + CONTEXT
    )
    with pytest.raises(ValueError, match="got 1 states for 3 blocks"):
        m(prompt, m.init_states(1, 16)[:1])


def test_generate_stops_at_eos_and_skips_the_forward_pass_nobody_reads():
    m = model()
    prompt = torch.randint(0, VOCAB, (1, 5))
    first = generate(m, prompt, 1, max_context=CONTEXT, temperature=0)[0, -1].item()
    assert generate(m, prompt, 40, max_context=CONTEXT, temperature=0, eos_id=first).shape == (1, 6)

    calls = []
    hook = m.register_forward_hook(lambda *_: calls.append(1))
    generate(m, prompt, 8, max_context=CONTEXT, temperature=0)
    hook.remove()
    assert len(calls) == 8  # one prefill + seven decodes: the eighth token needs no logits
