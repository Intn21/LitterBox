"""``score`` and ``confidence`` from a probability vector, and the packing into answers.

The formulas are pinned to TypeSafe's documented worked example to the digit,
then to their defining properties: uniform scores 0, a point mass scores 1,
the same top probability means more over more options, and mass near the mode
costs less than mass far from it.
"""

import math
import random

import pytest
import torch
from pydantic import ValidationError

from litterbox.decide import ChoiceAnswer, NoulAnswer, Request, ScoreAnswer
from litterbox.decide.derive import (
    answer,
    argmax,
    confidence_choice,
    confidence_score,
    respond,
    score_value,
)

SEVERITY = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
REQUEST = Request.model_validate(
    {
        "model": "jev-latest",
        "state": "The export button crashes the settings page in Safari.",
        "questions": {
            "severity": {"type": "score", "instructions": "How severe?", "criteria": SEVERITY},
            "dept": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": "", "technical": "", "other": ""},
            },
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        },
    }
)


# --- the documented example ---------------------------------------------------


def test_score_and_confidence_match_the_docs_example():
    p = [0.0, 0.57, 0.43]
    assert score_value(p) == pytest.approx(1.43)
    assert confidence_score(p) == pytest.approx(0.355, abs=1e-3)  # the docs print 0.35


def test_choice_confidence_matches_the_hand_computation():
    assert confidence_choice([0.81, 0.12, 0.07]) == pytest.approx((0.81 - 1 / 3) / (1 - 1 / 3))


# --- defining properties ------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 7, 255])
def test_uniform_is_zero_and_a_point_mass_is_one(n):
    uniform = [1 / n] * n
    point = [0.0] * n
    point[n // 2] = 1.0
    assert confidence_choice(uniform) == pytest.approx(0.0, abs=1e-9)
    assert confidence_choice(point) == pytest.approx(1.0)
    if n <= 10:
        assert confidence_score(uniform) == pytest.approx(0.0, abs=1e-9)
        assert confidence_score(point) == pytest.approx(1.0)


def test_the_same_top_probability_means_more_over_more_options():
    three = [0.81, 0.12, 0.07]
    ten = [0.81] + [0.19 / 9] * 9
    assert confidence_choice(ten) > confidence_choice(three)


def test_mass_next_to_the_mode_costs_less_than_mass_at_the_far_end():
    near = [0.0, 0.57, 0.43]  # 0.43 one step from the mode
    far = [0.43, 0.57, 0.0]  # the same 0.43 at the other end: still one step away
    farther = [0.43, 0.0, 0.57]  # two steps away
    assert confidence_score(near) == pytest.approx(confidence_score(far))
    assert confidence_score(farther) < confidence_score(near)


def test_score_is_the_expected_level_and_stays_in_range():
    assert score_value([1.0, 0.0, 0.0]) == 0.0
    assert score_value([0.0, 0.0, 1.0]) == 2.0
    assert score_value([0.25, 0.5, 0.25]) == pytest.approx(1.0)


def test_ties_go_to_the_first_listed_option():
    assert argmax([0.4, 0.4, 0.2]) == 0
    assert answer(REQUEST.questions["dept"], [0.4, 0.4, 0.2]).choice == "billing"


def test_float_noise_is_clamped_not_rejected():
    p = [1.0 + 1e-7, 0.0, 0.0]
    assert confidence_choice(p) == 1.0
    assert confidence_score(p) == 1.0
    assert answer(REQUEST.questions["dept"], p).confidence == 1.0


def test_accepts_tensors_and_numpy_as_well_as_lists():
    p = torch.tensor([0.0, 0.57, 0.43])
    assert score_value(p) == pytest.approx(1.43)
    assert confidence_score(p.numpy()) == pytest.approx(confidence_score(p.tolist()))


@pytest.mark.parametrize("fn", [confidence_choice, confidence_score])
def test_confidence_needs_two_answers(fn):
    with pytest.raises(ValueError, match="at least two"):
        fn([1.0])


# --- packing ------------------------------------------------------------------


def test_answer_packs_each_question_type():
    dept = answer(REQUEST.questions["dept"], [0.05, 0.9, 0.05])
    assert isinstance(dept, ChoiceAnswer)
    assert dept.choice == "technical"
    assert dept.probabilities == {"billing": 0.05, "technical": 0.9, "other": 0.05}

    sev = answer(REQUEST.questions["severity"], [0.0, 0.57, 0.43])
    assert isinstance(sev, ScoreAnswer)
    assert sev.score == pytest.approx(1.43)
    assert sev.legend == dict(enumerate(SEVERITY))
    assert sev.probabilities == {0: 0.0, 1: 0.57, 2: 0.43}

    angry = answer(REQUEST.questions["angry"], [0.88, 0.12])
    assert isinstance(angry, NoulAnswer)
    assert angry.noul == pytest.approx(0.12)


def test_answer_refuses_a_vector_of_the_wrong_length():
    with pytest.raises(ValueError, match="3 options but 4 probabilities"):
        answer(REQUEST.questions["dept"], [0.25] * 4)
    with pytest.raises(ValueError, match="3 levels but 2 probabilities"):
        answer(REQUEST.questions["severity"], [0.5, 0.5])
    with pytest.raises(ValueError, match="two answers"):
        answer(REQUEST.questions["angry"], [0.2, 0.3, 0.5])


def test_answer_refuses_a_vector_that_is_not_a_distribution():
    with pytest.raises(ValidationError, match="sum to 1"):
        answer(REQUEST.questions["dept"], [0.5, 0.5, 0.5])


def test_respond_builds_a_checked_response():
    resp = respond(
        REQUEST,
        {"severity": [0.0, 0.57, 0.43], "dept": [0.05, 0.9, 0.05], "angry": [0.88, 0.12]},
        model="litterbox-decide",
        input_tokens=42,
    )
    assert set(resp.answers) == {"severity", "dept", "angry"}
    assert resp.usage.input_tokens == 42 and resp.usage.output_tokens == 0
    resp.check_against(REQUEST)


def test_respond_refuses_missing_or_extra_questions():
    with pytest.raises(ValueError, match="missing for \\['angry'\\]"):
        respond(REQUEST, {"severity": [1, 0, 0], "dept": [1, 0, 0]}, model="m", input_tokens=1)
    with pytest.raises(ValueError, match="extra for \\['mood'\\]"):
        respond(
            REQUEST,
            {"severity": [1, 0, 0], "dept": [1, 0, 0], "angry": [1, 0], "mood": [1, 0]},
            model="m",
            input_tokens=1,
        )


# --- the closure property -------------------------------------------------------


def random_request(rng):
    questions = {}
    for i in range(rng.randint(1, 6)):
        kind = rng.choice(["noul", "choice", "score"])
        if kind == "noul":
            q = {"type": "noul", "instructions": "?"}
        elif kind == "choice":
            n = rng.randint(2, 255)
            q = {
                "type": "choice",
                "instructions": "?",
                "criteria": {f"opt{j}": "" for j in range(n)},
            }
        else:
            q = {
                "type": "score",
                "instructions": "?",
                "criteria": [f"l{j}" for j in range(rng.randint(2, 10))],
            }
        questions[f"q{i}"] = q
    return Request.model_validate({"model": "m", "state": "s", "questions": questions})


def random_softmax(rng, n):
    logits = [rng.gauss(0, 3) for _ in range(n)]
    top = max(logits)
    e = [math.exp(z - top) for z in logits]
    total = sum(e)
    return [v / total for v in e]


def test_any_valid_request_with_any_softmax_vector_yields_an_in_schema_response():
    rng = random.Random(0)
    for _ in range(300):
        req = random_request(rng)
        sizes = {
            qid: 2 if q.type == "noul" else len(q.criteria) for qid, q in req.questions.items()
        }
        probs = {qid: random_softmax(rng, n) for qid, n in sizes.items()}
        resp = respond(req, probs, model="m", input_tokens=1)  # raises if anything is off
        for qid, a in resp.answers.items():
            assert a.type == req.questions[qid].type
            if isinstance(a, ChoiceAnswer):
                assert a.choice in req.questions[qid].criteria
                assert 0.0 <= a.confidence <= 1.0
            if isinstance(a, ScoreAnswer):
                assert 0.0 <= a.score <= len(req.questions[qid].criteria) - 1
                assert 0.0 <= a.confidence <= 1.0
