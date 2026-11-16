"""One decision example: a state, its questions, the drawn labels, and the truth.

This is the single format every dataset in the decision-model work uses, one
JSON object per line. ``state`` and ``questions`` are exactly what a request
carries, so training and evaluation read the same file the readout does.

``labels`` holds, per question, the index of the answer in the order the
question lists them: an option for a ``choice``, a level for a ``score``, and
0 = no / 1 = yes for a ``noul``. Training reads this and nothing below it.

``truth`` is the distribution each label was drawn from, in the same order,
and ``region`` names the rule that set it as ``kind/name``. Generated data has
both: they are the answer key, read by evaluation and never by training.
Public data has no region, and has a truth only where the source recorded how
its annotators split.

``group`` ties together records that must not be separated by a split: the
same state asked several questions, or paraphrases of one item.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from litterbox.decide.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    Request,
    ScoreQuestion,
    Text,
)


def n_answers(question: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> int:
    """How many answers a question has; a noul has two, no and yes."""
    if isinstance(question, NoulQuestion):
        return 2
    return len(question.criteria)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    state: Text
    questions: dict[str, Question]
    labels: dict[str, int]
    truth: dict[str, list[float]] | None = None
    region: dict[str, str] | None = None
    group: str | None = None  # records sharing a group must stay in one split
    license: str | None = None  # the source's usage class, where the source states one

    @model_validator(mode="after")
    def _consistent(self) -> Record:
        qids = set(self.questions)
        if set(self.labels) != qids:
            raise ValueError("labels must cover exactly the questions")
        for qid, q in self.questions.items():
            n = n_answers(q)
            if not 0 <= self.labels[qid] < n:
                raise ValueError(f"{qid!r}: label {self.labels[qid]} is not one of {n} answers")
        if self.truth is not None:
            if set(self.truth) != qids:
                raise ValueError("truth must cover exactly the questions")
            for qid, dist in self.truth.items():
                if len(dist) != n_answers(self.questions[qid]):
                    raise ValueError(f"{qid!r}: truth has the wrong number of entries")
                if min(dist) < 0 or not math.isclose(sum(dist), 1.0, abs_tol=1e-6):
                    raise ValueError(f"{qid!r}: truth is not a distribution")
        if self.region is not None and set(self.region) != qids:
            raise ValueError("region must cover exactly the questions")
        return self

    def request(self, model: str = "litterbox") -> Request:
        """This record as an API request, for the readout or the server."""
        return Request(model=model, state=self.state, questions=self.questions)

    def kind(self, qid: str) -> str | None:
        """``deterministic``, ``stochastic`` or ``unrecoverable``; ``None`` for public data."""
        return None if self.region is None else self.region[qid].split("/", 1)[0]


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    n = 0
    with Path(path).open("w") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield Record.model_validate_json(line)
