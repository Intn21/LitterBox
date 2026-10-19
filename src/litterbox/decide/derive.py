"""From a probability vector to an answer: ``score``, ``confidence``, and the packing.

The model's only output is a probability per allowed answer. Everything else
the API returns is arithmetic on that vector, and it is done here so that any
readout, the label-token baseline or the pointer head, produces identical
answers from identical probabilities.

Three formulas, each matching TypeSafe's documentation and its worked example
(``docs/decision-model.md`` has the derivation, the demo notebook the trace):

- ``score`` is the probability-weighted level index, ``0·0.00 + 1·0.57 + 2·0.43
  = 1.43``, fractional on purpose.
- a ``choice``'s ``confidence`` is the top probability with chance subtracted
  and the result stretched to 0..1: ``(p_max − 1/n) / (1 − 1/n)``. The same
  ``p_max`` means more over ten options than over three, and a threshold in
  software needs one scale.
- a ``score``'s ``confidence`` measures distance rather than height: one minus
  the expected distance from the modal level, relative to that distance under
  a uniform distribution. Mass on a neighbouring level costs less than mass at
  the far end, which is what "ordered" buys.

A ``noul`` derives nothing: with two answers, the probability of *yes* already
says how sure the model is.

None of this is where honesty comes from. If the probabilities are calibrated
the confidence is; if they are not, this reports the dishonest number
faithfully. Temperature scaling acts on the scores before the softmax, upstream
of here, and the calibration module measures the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from litterbox.decide.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)


def _as_floats(p: Sequence[float]) -> list[float]:
    # Accepts a list, a tuple, a numpy array or a torch tensor without importing either.
    values = p.tolist() if hasattr(p, "tolist") else list(p)
    return [float(v) for v in values]


def _clamp(x: float) -> float:
    # A softmax that sums to 1 + 1e-7 can push a formula a hair past its range.
    return min(1.0, max(0.0, x))


def argmax(p: Sequence[float]) -> int:
    """Index of the largest entry; ties go to the first listed, deterministically."""
    p = _as_floats(p)
    return max(range(len(p)), key=p.__getitem__)


def confidence_choice(p: Sequence[float]) -> float:
    """How far the top probability sits above an even split: 0 for uniform, 1 for a point mass."""
    p = _as_floats(p)
    n = len(p)
    if n < 2:
        raise ValueError("confidence needs at least two options")
    return _clamp((max(p) - 1 / n) / (1 - 1 / n))


def score_value(p: Sequence[float]) -> float:
    """Probability-weighted level index, 0 to ``n − 1``; fractional on purpose."""
    p = _as_floats(p)
    return _clamp_score(sum(i * pi for i, pi in enumerate(p)), len(p))


def _clamp_score(s: float, n: int) -> float:
    return min(float(n - 1), max(0.0, s))


def confidence_score(p: Sequence[float]) -> float:
    """One minus the expected distance from the modal level, relative to uniform.

    ``[0.0, 0.57, 0.43]``: mode 1, expected distance ``0.43``, uniform distance
    ``(1 + 0 + 1) / 3 = 0.667``, confidence ``1 − 0.43 / 0.667 = 0.355``.
    """
    p = _as_floats(p)
    n = len(p)
    if n < 2:
        raise ValueError("confidence needs at least two levels")
    m = argmax(p)
    distance = sum(pi * abs(i - m) for i, pi in enumerate(p))
    uniform = sum(abs(i - m) for i in range(n)) / n
    return _clamp(1 - distance / uniform)


def answer(question: NoulQuestion | ChoiceQuestion | ScoreQuestion, p: Sequence[float]) -> Answer:
    """Pack a probability vector over ``question``'s answers into the matching answer model.

    ``p`` is ordered as the question listed its options or levels. Its length
    must match; that is the first place a readout bug shows. The answer model's
    own validators then re-check that ``p`` sums to one and that the named
    choice is the largest entry, so nothing here trusts the readout.
    """
    p = _as_floats(p)
    if isinstance(question, ChoiceQuestion):
        options = question.options
        if len(p) != len(options):
            raise ValueError(f"{len(options)} options but {len(p)} probabilities")
        return ChoiceAnswer(
            type="choice",
            choice=options[argmax(p)],
            probabilities=dict(zip(options, p, strict=True)),
            confidence=confidence_choice(p),
        )
    if isinstance(question, ScoreQuestion):
        if len(p) != question.levels:
            raise ValueError(f"{question.levels} levels but {len(p)} probabilities")
        return ScoreAnswer(
            type="score",
            score=score_value(p),
            legend=dict(enumerate(question.criteria)),
            probabilities=dict(enumerate(p)),
            confidence=confidence_score(p),
        )
    if isinstance(question, NoulQuestion):
        if len(p) != 2:
            raise ValueError(f"a noul has two answers, no and yes, but got {len(p)} probabilities")
        return NoulAnswer(type="noul", noul=_clamp(p[1]))
    raise TypeError(f"not a question: {type(question).__name__}")


def respond(
    request: Request,
    probabilities: Mapping[str, Sequence[float]],
    *,
    model: str,
    input_tokens: int,
) -> Response:
    """Build the full response for ``request`` from one probability vector per question id.

    The result is checked against the request before it is returned, so a
    response that answers an unasked question or names an unoffered option
    cannot leave this function.
    """
    missing = set(request.questions) - set(probabilities)
    extra = set(probabilities) - set(request.questions)
    if missing or extra:
        raise ValueError(f"probabilities missing for {sorted(missing)}, extra for {sorted(extra)}")
    response = Response(
        model=model,
        answers={qid: answer(q, probabilities[qid]) for qid, q in request.questions.items()},
        usage=Usage(input_tokens=input_tokens, output_tokens=0),
    )
    response.check_against(request)
    return response
