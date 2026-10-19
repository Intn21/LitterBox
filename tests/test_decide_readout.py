"""The baseline readout: lettered prompts, one pass, the letters' logits.

Most of these tests drive the readout with a hand-written ``logits_fn`` whose
behaviour is known exactly, so that the prompt format, the label masking and
the order mapping are pinned independently of any model. The last few run a
real (random) litterbox transformer end to end, which is what proves the
batching and the schema closure.
"""

import random

import pytest
import torch

from litterbox.data.tokenizer import Tokenizer
from litterbox.data.tokenizer.byte import CharTokenizer
from litterbox.decide import Request
from litterbox.decide.readout import (
    ALPHABET,
    PREAMBLE,
    decide,
    encode_prompts,
    hf_logits,
    label_ids,
    probabilities,
    render,
)
from litterbox.model import dense_transformer

REQUEST = Request.model_validate(
    {
        "model": "jev-latest",
        "state": "I was charged twice this month and nobody answers the phone.",
        "questions": {
            "dept": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {
                    "billing": "payments, invoices",
                    "technical": "bugs, outages",
                    "other": "",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is it?",
                "criteria": ["can wait", "this week", "today"],
            },
            "angry": {
                "type": "noul",
                "instructions": "Is the customer angry?",
                "criteria": {"true": "shouting", "false": "calm"},
            },
        },
    }
)

TOK = CharTokenizer()  # one token per character: every letter is a single token, no network
LABEL = {c: TOK.encode(c)[0] for c in ALPHABET}


def favour(letters_by_prompt):
    """A ``logits_fn`` that puts a big logit on one letter per prompt, at the last position."""

    def fn(ids):
        logits = torch.zeros(ids.shape[0], ids.shape[1], TOK.vocab_size)
        for i, letter in enumerate(letters_by_prompt):
            logits[i, :, LABEL[letter]] = 5.0
        return logits

    return fn


# --- rendering ---------------------------------------------------------------------


def test_render_pins_the_prompt_format():
    assert render(REQUEST, "dept") == "\n".join(
        [
            PREAMBLE,
            "",
            "State:",
            "I was charged twice this month and nobody answers the phone.",
            "",
            "Question: Which department should handle this?",
            "A = billing: payments, invoices",
            "B = technical: bugs, outages",
            "C = other",
            "Answer:",
        ]
    )


def test_render_noul_as_no_then_yes_and_score_as_ordered_levels():
    angry = render(REQUEST, "angry")
    assert "A = no: calm\nB = yes: shouting\nAnswer:" in angry
    urgency = render(REQUEST, "urgency")
    assert "A = level 0: can wait\nB = level 1: this week\nC = level 2: today\nAnswer:" in urgency


def test_render_shows_options_in_the_given_order():
    prompt = render(REQUEST, "dept", order=[2, 0, 1])
    assert "A = other\nB = billing: payments, invoices\nC = technical: bugs, outages" in prompt


def test_render_dumps_json_state_and_structured_instructions():
    req = Request.model_validate(
        {
            "model": "m",
            "state": {"ticket": {"text": "hi"}},
            "questions": {"q": {"type": "noul", "instructions": {"ask": "is it polite?"}}},
        }
    )
    prompt = render(req, "q")
    assert '"ticket": {\n    "text": "hi"\n  }' in prompt
    assert 'Question: {\n  "ask": "is it polite?"\n}' in prompt


def test_render_refuses_more_options_than_labels():
    req = Request.model_validate(
        {
            "model": "m",
            "state": "s",
            "questions": {
                "wide": {
                    "type": "choice",
                    "instructions": "?",
                    "criteria": {f"o{i}": "" for i in range(63)},
                }
            },
        }
    )
    with pytest.raises(ValueError, match="63 options; this readout has 62 labels"):
        render(req, "wide")


# --- tokens --------------------------------------------------------------------------


def test_label_ids_prefers_the_spaced_form_and_falls_back_to_bare_letters():
    assert label_ids(TOK, 3) == [LABEL["A"], LABEL["B"], LABEL["C"]]  # ' A' is two chars here

    class Spaced(Tokenizer):
        name, vocab_size = "spaced", 1000

        def encode(self, text):
            return [ord(text[-1]) + 500] if text.startswith(" ") else [1, 2]

        def decode(self, ids):
            return ""

    assert label_ids(Spaced(), 2) == [ord("A") + 500, ord("B") + 500]


def test_label_ids_refuses_a_tokenizer_with_multi_token_letters():
    class Pairs(Tokenizer):
        name, vocab_size = "pairs", 1000

        def encode(self, text):
            return [1, 2]

        def decode(self, ids):
            return ""

    with pytest.raises(ValueError, match="not single tokens"):
        label_ids(Pairs(), 2)


def test_gpt2_letters_are_single_tokens_with_a_leading_space():
    pytest.importorskip("tiktoken")
    from litterbox.data.tokenizer import build_tokenizer

    gpt2 = build_tokenizer({"type": "tiktoken", "encoding": "gpt2"})
    ids = label_ids(gpt2, 62)
    assert ids[:2] == [317, 347]  # ' A', ' B'
    assert len(set(ids)) == 62


def test_encode_prompts_right_pads_and_records_each_last_position():
    ids, last = encode_prompts(TOK, ["ab", "abcd"], pad_id=0)
    assert ids.shape == (2, 4)
    assert ids[0, 2:].tolist() == [0, 0]
    assert last.tolist() == [1, 3]


# --- the readout with a known model ----------------------------------------------------


def test_probabilities_follow_the_favoured_letter_and_come_back_in_request_order():
    probs, tokens = probabilities(favour(["B", "C", "A"]), TOK, REQUEST)
    assert probs["dept"].index(max(probs["dept"])) == 1  # B = technical
    assert probs["urgency"].index(max(probs["urgency"])) == 2  # C = today
    assert probs["angry"].index(max(probs["angry"])) == 0  # A = no
    assert all(abs(sum(p) - 1) < 1e-6 for p in probs.values())
    assert tokens == sum(len(TOK.encode(render(REQUEST, q))) for q in REQUEST.questions)


def test_labels_beyond_a_questions_options_are_never_considered():
    # A huge logit on 'D' must not leak into a three-option question: only A, B, C exist for it.
    probs, _ = probabilities(favour(["D", "D", "D"]), TOK, REQUEST)
    for p in probs.values():
        assert len(p) == 3 or len(p) == 2
        assert all(abs(v - 1 / len(p)) < 1e-6 for v in p)  # the rest tie exactly


def test_a_shown_order_is_mapped_back_to_the_requests_order():
    # The model always favours the first letter; shown as [other, billing, technical], 'A' is other.
    probs, _ = probabilities(favour(["A", "A", "A"]), TOK, REQUEST, orders={"dept": [2, 0, 1]})
    assert probs["dept"].index(max(probs["dept"])) == 2  # 'other' is index 2 in the request
    assert probs["urgency"].index(max(probs["urgency"])) == 0  # unshuffled question unaffected


# --- the readout with a real (random) transformer ------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return dense_transformer(
        TOK.vocab_size, d_model=32, n_layers=2, heads=2, max_seq_len=1024
    ).eval()


def test_decide_on_the_repo_backbone_returns_a_checked_response(tiny):
    resp = decide(tiny, TOK, REQUEST)
    resp.check_against(REQUEST)
    assert set(resp.answers) == {"dept", "urgency", "angry"}
    assert resp.usage.output_tokens == 0 and resp.usage.input_tokens > 0


def test_batching_does_not_change_the_numbers(tiny):
    one, _ = probabilities(tiny, TOK, REQUEST, batch_size=1)
    many, _ = probabilities(tiny, TOK, REQUEST, batch_size=16)
    for qid in REQUEST.questions:
        assert one[qid] == pytest.approx(many[qid], abs=1e-5)


def test_schema_closure_holds_for_random_requests_on_a_random_model(tiny):
    rng = random.Random(0)
    for _ in range(25):
        questions = {}
        for i in range(rng.randint(1, 4)):
            kind = rng.choice(["noul", "choice", "score"])
            if kind == "noul":
                q = {"type": "noul", "instructions": rng.choice(["ok?", "angry?"])}
            elif kind == "choice":
                q = {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {f"o{j}": "" for j in range(rng.randint(2, 12))},
                }
            else:
                q = {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": [f"l{j}" for j in range(rng.randint(2, 10))],
                }
            questions[f"q{i}"] = q
        req = Request.model_validate(
            {"model": "m", "state": "x" * rng.randint(1, 40), "questions": questions}
        )
        decide(tiny, TOK, req).check_against(req)  # raises on any out-of-schema answer


def test_hf_logits_wraps_a_transformers_style_model():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            assert attention_mask.shape == input_ids.shape and bool(attention_mask.all())
            out = torch.zeros(*input_ids.shape, TOK.vocab_size)
            out[..., LABEL["C"]] = 3.0
            return type("Out", (), {"logits": out})()

    probs, _ = probabilities(hf_logits(Fake()), TOK, REQUEST)
    assert probs["dept"].index(max(probs["dept"])) == 2
