"""Decision models: typed questions in, calibrated probabilities out, one pass.

See ``docs/decision-model.md`` for what this is and the roadmap for building
it, and ``demo/decide/decision_model.ipynb`` for the mechanism traced by hand.
"""

from litterbox.decide.calibration import (
    aurc,
    brier,
    coverage_at,
    ece,
    ece_noise_floor,
    log_loss,
    reliability,
    summarize,
)
from litterbox.decide.derive import (
    answer,
    confidence_choice,
    confidence_score,
    respond,
    score_value,
)
from litterbox.decide.readout import decide, hf_logits, render
from litterbox.decide.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "Request",
    "Response",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
    "answer",
    "aurc",
    "brier",
    "confidence_choice",
    "confidence_score",
    "coverage_at",
    "decide",
    "ece",
    "ece_noise_floor",
    "hf_logits",
    "log_loss",
    "reliability",
    "render",
    "respond",
    "score_value",
    "summarize",
]
