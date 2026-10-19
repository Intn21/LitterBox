"""The decision API's request and response shapes.

A decision model answers typed questions about a piece of state and returns
probabilities over answers it was given, never prose. This module is the
contract for that exchange: what a request may contain, what a response must
contain, and the limits on both. It computes nothing. The serializer turns a
:class:`Request` into tokens, the readout produces probability vectors, and
:mod:`litterbox.decide.derive` turns those into the answer models here.

The field names are TypeSafe's (``noul``, ``criteria``, ``legend``), odd as
some of them are, because the roadmap's exit condition is that their SDKs and
the public benchmark repositories run against our server unchanged. Cloudflare's
Clef and Alibaba's decision model made the same choice for the same reason.

Validation is strict, like :mod:`litterbox.utils.config`: an unknown key, a
``choice`` with 300 options or a ``score`` with one level fails at the door
with a message naming the field, instead of surfacing as a shape error inside a
forward pass. The limits are the documented ones, 255 options and 2 to 10
levels, plus one of ours: a ``choice`` needs at least two options, because with
one there is nothing to decide and the confidence formula divides by zero.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

# Probabilities are softmax outputs serialised through JSON; they sum to one up
# to float error, and a response that is off by more than this is a bug.
PROBABILITY_TOLERANCE = 1e-4

# Instructions, option descriptions and level descriptions may be plain text
# or JSON structure; the docs' "advanced structure" page allows both.
Text = str | dict[str, Any] | list[Any]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- questions --------------------------------------------------------------


class NoulQuestion(_Strict):
    """Is this true? Answered with one number, the probability of *yes*."""

    type: Literal["noul"]
    instructions: Text
    criteria: dict[Literal["true", "false"], Text] | None = None


class ChoiceQuestion(_Strict):
    """Which of these? ``criteria`` maps each option's name to its description."""

    type: Literal["choice"]
    instructions: Text
    criteria: dict[str, Text]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, criteria: dict[str, Text]) -> dict[str, Text]:
        n = len(criteria)
        if not MIN_CHOICE_OPTIONS <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"a choice needs {MIN_CHOICE_OPTIONS} to {MAX_CHOICE_OPTIONS} options, got {n}"
            )
        if any(not name for name in criteria):
            raise ValueError("option names must be non-empty strings")
        return criteria

    @property
    def options(self) -> list[str]:
        return list(self.criteria)


class ScoreQuestion(_Strict):
    """Where on this ordered scale? ``criteria`` lists the levels, lowest first."""

    type: Literal["score"]
    instructions: Text
    criteria: list[Text]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, criteria: list[Text]) -> list[Text]:
        n = len(criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {n}"
            )
        return criteria

    @property
    def levels(self) -> int:
        return len(self.criteria)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class Request(_Strict):
    """One state and a map of named, typed questions about it.

    Question ids are the caller's; they come back as the keys of the answer map.
    Every question sees the same state and none sees another question, so the
    map's order carries no meaning.
    """

    model: str
    state: Text
    questions: dict[str, Question]

    @field_validator("questions")
    @classmethod
    def _at_least_one(cls, questions: dict[str, Question]) -> dict[str, Question]:
        if not questions:
            raise ValueError("a request needs at least one question")
        if any(not qid for qid in questions):
            raise ValueError("question ids must be non-empty strings")
        return questions


# --- answers ----------------------------------------------------------------


def _check_distribution(probabilities: dict[Any, float], what: str) -> None:
    if any(p < 0 for p in probabilities.values()):
        raise ValueError(f"{what} must be non-negative")
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, abs_tol=PROBABILITY_TOLERANCE):
        raise ValueError(f"{what} must sum to 1, got {total:.6f}")


class NoulAnswer(_Strict):
    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(_Strict):
    """The top option by name, the full distribution over options, and a confidence."""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> ChoiceAnswer:
        _check_distribution(self.probabilities, "choice probabilities")
        if self.choice not in self.probabilities:
            raise ValueError(f"choice {self.choice!r} is not one of the options")
        top = max(self.probabilities.values())
        if self.probabilities[self.choice] < top - PROBABILITY_TOLERANCE:
            raise ValueError(f"choice {self.choice!r} is not the most probable option")
        return self


class ScoreAnswer(_Strict):
    """A probability per level, the probability-weighted level as ``score``, and a confidence.

    Levels are numbered from 0 in the order the request listed them; ``legend``
    maps each number back to its description so the score can be read.
    """

    type: Literal["score"]
    score: float
    legend: dict[int, Text]
    probabilities: dict[int, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> ScoreAnswer:
        _check_distribution(self.probabilities, "score probabilities")
        n = len(self.probabilities)
        if set(self.probabilities) != set(range(n)):
            raise ValueError(f"score levels must be numbered 0 to {n - 1} with none missing")
        if set(self.legend) != set(self.probabilities):
            raise ValueError("legend and probabilities must cover the same levels")
        if not -PROBABILITY_TOLERANCE <= self.score <= n - 1 + PROBABILITY_TOLERANCE:
            raise ValueError(f"score must lie between 0 and {n - 1}, got {self.score}")
        return self


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(_Strict):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Response(_Strict):
    model: str
    answers: dict[str, Answer]
    usage: Usage

    def check_against(self, request: Request) -> None:
        """Raise ``ValueError`` unless this response answers exactly ``request``.

        Each question id must be answered once, with the matching answer type,
        over exactly the options or levels the request supplied. This is what
        the fuzz test asserts: whatever the readout does, it cannot answer a
        question that was not asked or name an option that was not offered.
        """
        asked, answered = set(request.questions), set(self.answers)
        if asked != answered:
            raise ValueError(
                f"answered {sorted(answered - asked)} that were not asked; "
                f"left {sorted(asked - answered)} unanswered"
            )
        for qid, question in request.questions.items():
            answer = self.answers[qid]
            if answer.type != question.type:
                raise ValueError(f"{qid!r}: asked a {question.type}, answered a {answer.type}")
            if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
                if set(answer.probabilities) != set(question.criteria):
                    raise ValueError(f"{qid!r}: answer options do not match the question's")
            if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
                if len(answer.probabilities) != question.levels:
                    raise ValueError(f"{qid!r}: answer has a different number of levels")
                if list(answer.legend[i] for i in range(question.levels)) != question.criteria:
                    raise ValueError(f"{qid!r}: legend does not match the question's levels")
