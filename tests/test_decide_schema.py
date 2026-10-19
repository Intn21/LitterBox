"""The decision API contract: what a request may say and what a response must.

Nothing here runs a model. These tests pin the shapes and limits that the
serializer, the readout and the server all build on, and the one cross-check,
``Response.check_against``, that the later fuzz test uses to prove schema
closure.
"""

import pytest
from pydantic import ValidationError

from litterbox.decide import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
)

# The worked example from docs.typesafe.ai/primitives/score, plus a choice and a noul.
TICKET = "The export button crashes the settings page in Safari. It works in Chrome."
SEVERITY = [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists",
]
REQUEST = {
    "model": "jev-latest",
    "state": TICKET,
    "questions": {
        "bug_severity": {
            "type": "score",
            "instructions": "How severe is the reported issue?",
            "criteria": SEVERITY,
        },
        "dept": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {
                "billing": "payments, invoices",
                "technical": "bugs, outages",
                "other": "",
            },
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
    },
}
RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "bug_severity": {
            "type": "score",
            "score": 1.43,
            "confidence": 0.35,
            "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
            "legend": {"0": SEVERITY[0], "1": SEVERITY[1], "2": SEVERITY[2]},
        },
        "dept": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.05, "technical": 0.9, "other": 0.05},
            "confidence": 0.85,
        },
        "angry": {"type": "noul", "noul": 0.12},
    },
    "usage": {"input_tokens": 120, "output_tokens": 0},
}


def choice(n):
    return {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": "" for i in range(n)}}


def score(n):
    return {"type": "score", "instructions": "rate", "criteria": [f"level {i}" for i in range(n)]}


def request(**questions):
    return Request.model_validate({"model": "jev-latest", "state": "x", "questions": questions})


# --- requests ---------------------------------------------------------------


def test_docs_example_parses_with_the_right_types():
    req = Request.model_validate(REQUEST)
    assert isinstance(req.questions["bug_severity"], ScoreQuestion)
    assert isinstance(req.questions["dept"], ChoiceQuestion)
    assert req.questions["dept"].options == ["billing", "technical", "other"]
    assert req.questions["bug_severity"].levels == 3


def test_state_may_be_text_or_json():
    for state in ("plain text", {"ticket": {"text": "x"}}, ["a", "b"]):
        assert Request.model_validate({**REQUEST, "state": state}).state == state


def test_instructions_and_descriptions_may_be_structured():
    q = {
        "type": "choice",
        "instructions": {"task": "route", "fields": ["a"]},
        "criteria": {"x": ["one", "two"], "y": {"k": "v"}},
    }
    assert isinstance(request(q=q).questions["q"], ChoiceQuestion)


@pytest.mark.parametrize("n", [2, 255])
def test_choice_option_limits_are_inclusive(n):
    assert len(request(q=choice(n)).questions["q"].criteria) == n


@pytest.mark.parametrize("n", [0, 1, 256])
def test_choice_rejects_too_few_or_too_many_options(n):
    with pytest.raises(ValidationError, match="2 to 255 options"):
        request(q=choice(n))


@pytest.mark.parametrize("n", [2, 10])
def test_score_level_limits_are_inclusive(n):
    assert request(q=score(n)).questions["q"].levels == n


@pytest.mark.parametrize("n", [0, 1, 11])
def test_score_rejects_too_few_or_too_many_levels(n):
    with pytest.raises(ValidationError, match="2 to 10 levels"):
        request(q=score(n))


def test_noul_criteria_are_optional_and_keyed_true_false():
    q = {
        "type": "noul",
        "instructions": "angry?",
        "criteria": {"true": "shouting", "false": "calm"},
    }
    assert request(q=q).questions["q"].criteria["true"] == "shouting"
    with pytest.raises(ValidationError):
        request(q={**q, "criteria": {"yes": "shouting"}})


def test_unknown_question_type_is_rejected():
    with pytest.raises(ValidationError, match="type"):
        request(q={"type": "rank", "instructions": "order these", "criteria": ["a", "b"]})


def test_unknown_keys_are_rejected_everywhere():
    with pytest.raises(ValidationError, match="extra"):
        Request.model_validate({**REQUEST, "temperature": 0.7})
    with pytest.raises(ValidationError, match="extra"):
        request(q={**choice(2), "max_tokens": 5})


def test_missing_instructions_is_rejected():
    with pytest.raises(ValidationError, match="instructions"):
        request(q={"type": "choice", "criteria": {"a": "", "b": ""}})


def test_empty_question_map_and_empty_ids_are_rejected():
    with pytest.raises(ValidationError, match="at least one question"):
        Request.model_validate({**REQUEST, "questions": {}})
    with pytest.raises(ValidationError, match="non-empty"):
        request(**{"": choice(2)})


def test_empty_option_names_are_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        request(q={"type": "choice", "instructions": "pick", "criteria": {"": "", "a": ""}})


# --- answers ----------------------------------------------------------------


def test_docs_example_response_parses_and_matches_its_request():
    resp = Response.model_validate(RESPONSE)
    assert isinstance(resp.answers["bug_severity"], ScoreAnswer)
    assert resp.answers["bug_severity"].probabilities[1] == 0.57  # JSON string keys become ints
    assert isinstance(resp.answers["dept"], ChoiceAnswer)
    assert isinstance(resp.answers["angry"], NoulAnswer)
    resp.check_against(Request.model_validate(REQUEST))


def test_response_round_trips_through_json():
    resp = Response.model_validate(RESPONSE)
    again = Response.model_validate_json(resp.model_dump_json())
    assert again == resp


@pytest.mark.parametrize(
    "probabilities", [{"a": 0.6, "b": 0.6}, {"a": 0.5, "b": 0.3}, {"a": 1.2, "b": -0.2}]
)
def test_choice_probabilities_must_be_a_distribution(probabilities):
    with pytest.raises(ValidationError, match="probabilities"):
        ChoiceAnswer(type="choice", choice="a", probabilities=probabilities, confidence=0.5)


def test_choice_tolerates_float_error_in_the_sum():
    ChoiceAnswer(type="choice", choice="a", probabilities={"a": 0.70001, "b": 0.3}, confidence=0.4)


def test_choice_must_name_the_most_probable_option():
    with pytest.raises(ValidationError, match="not the most probable"):
        ChoiceAnswer(type="choice", choice="b", probabilities={"a": 0.7, "b": 0.3}, confidence=0.4)
    with pytest.raises(ValidationError, match="not one of the options"):
        ChoiceAnswer(type="choice", choice="c", probabilities={"a": 0.7, "b": 0.3}, confidence=0.4)


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_and_noul_are_bounded(confidence):
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            type="choice", choice="a", probabilities={"a": 1.0, "b": 0.0}, confidence=confidence
        )
    with pytest.raises(ValidationError):
        NoulAnswer(type="noul", noul=confidence)


def test_score_levels_must_be_contiguous_from_zero_and_match_the_legend():
    ok = dict(
        type="score",
        score=1.0,
        legend={0: "lo", 1: "mid", 2: "hi"},
        probabilities={0: 0.0, 1: 1.0, 2: 0.0},
        confidence=1.0,
    )
    ScoreAnswer(**ok)
    with pytest.raises(ValidationError, match="numbered 0 to"):
        ScoreAnswer(**{**ok, "probabilities": {1: 0.0, 2: 1.0, 3: 0.0}})
    with pytest.raises(ValidationError, match="legend"):
        ScoreAnswer(**{**ok, "legend": {0: "lo", 1: "mid"}})
    with pytest.raises(ValidationError, match="between 0 and 2"):
        ScoreAnswer(**{**ok, "score": 2.5})


# --- the cross-check --------------------------------------------------------


def test_check_against_catches_an_answer_to_a_question_that_was_not_asked():
    resp = Response.model_validate(RESPONSE)
    req = Request.model_validate(
        {**REQUEST, "questions": {k: v for k, v in REQUEST["questions"].items() if k != "angry"}}
    )
    with pytest.raises(ValueError, match="not asked"):
        resp.check_against(req)


def test_check_against_catches_an_unanswered_question():
    resp = Response.model_validate(
        {**RESPONSE, "answers": {k: v for k, v in RESPONSE["answers"].items() if k != "angry"}}
    )
    with pytest.raises(ValueError, match="unanswered"):
        resp.check_against(Request.model_validate(REQUEST))


def test_check_against_catches_an_option_that_was_not_offered():
    answers = dict(RESPONSE["answers"])
    answers["dept"] = {
        "type": "choice",
        "choice": "sales",
        "probabilities": {"sales": 0.9, "technical": 0.1},
        "confidence": 0.8,
    }
    with pytest.raises(ValueError, match="options do not match"):
        Response.model_validate({**RESPONSE, "answers": answers}).check_against(
            Request.model_validate(REQUEST)
        )


def test_check_against_catches_a_type_mismatch_and_a_wrong_legend():
    answers = dict(RESPONSE["answers"])
    answers["angry"] = {
        "type": "choice",
        "choice": "yes",
        "probabilities": {"yes": 1.0, "no": 0.0},
        "confidence": 1.0,
    }
    with pytest.raises(ValueError, match="asked a noul, answered a choice"):
        Response.model_validate({**RESPONSE, "answers": answers}).check_against(
            Request.model_validate(REQUEST)
        )
    answers = dict(RESPONSE["answers"])
    answers["bug_severity"] = {
        **RESPONSE["answers"]["bug_severity"],
        "legend": {"0": "a", "1": "b", "2": "c"},
    }
    with pytest.raises(ValueError, match="legend does not match"):
        Response.model_validate({**RESPONSE, "answers": answers}).check_against(
            Request.model_validate(REQUEST)
        )
