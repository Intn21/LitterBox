"""Where the label frequencies are written down: regions, questions, domains.

A *region* is one rule of a generated corpus: when this cue is in the state,
the label for this question is drawn from this distribution. Every frequency
in a corpus is a number in a :class:`Region`, so the whole design of a corpus
can be read in one place, and the generator records which region produced
each label.

Three kinds, and what an honest model should do in each:

- ``deterministic``: the cue fixes the label. Be certain.
- ``stochastic``: the cue leaves the label genuinely split, by a die we chose.
  Report the die.
- ``unrecoverable``: the label follows a cue that was rolled but never written
  into the state. Report the base rate. This kind is not declared here; it is
  what the questions named in :attr:`Domain.hide` become when a corpus is
  generated with ``unrecoverable=True``.

A cue is whatever the domain's ``render`` plants: a sentence for text domains,
a dict of fields for structured ones. A region with no cues is the "nothing
was said" case, and a question may have at most one, or two regions would be
indistinguishable and the recorded truth would be wrong.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from litterbox.decide.schema import Text

Cue = Any  # str for text domains, dict[str, Any] of fields for structured ones


def surfaces(cue: Cue) -> list[str]:
    """What a cue looks like in a state: the sentence itself, or its field names."""
    return [cue] if isinstance(cue, str) else list(cue)


@dataclass(frozen=True)
class Region:
    name: str
    kind: Literal["deterministic", "stochastic"]
    dist: tuple[float, ...]  # over the question's answers, in canonical order
    weight: float  # share of records that fall in this region
    cues: tuple[Cue, ...] = ()  # surface variants; one is planted per record


@dataclass(frozen=True)
class QuestionSpec:
    qid: str
    type: Literal["choice", "score", "noul"]
    instructions: tuple[str, ...]  # paraphrases; one is used per record
    regions: tuple[Region, ...]
    options: tuple[str, ...] = ()  # choice: option names; score: level descriptions

    @property
    def width(self) -> int:
        return 2 if self.type == "noul" else len(self.options)

    def base_rate(self) -> list[float]:
        """The regions mixed by weight: the honest answer when the cue is hidden."""
        return [sum(r.weight * r.dist[j] for r in self.regions) for j in range(self.width)]

    def as_question(self, instructions: str, order: list[int]) -> dict:
        """The schema question, with a choice's options listed in ``order``."""
        if self.type == "noul":
            return {"type": "noul", "instructions": instructions}
        if self.type == "score":
            return {"type": "score", "instructions": instructions, "criteria": list(self.options)}
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": {self.options[j]: "" for j in order},
        }

    def check(self) -> None:
        if not self.instructions or not self.regions:
            raise ValueError(f"{self.qid}: needs instructions and at least one region")
        if not math.isclose(sum(r.weight for r in self.regions), 1.0, abs_tol=1e-9):
            raise ValueError(f"{self.qid}: region weights must sum to 1")
        if sum(1 for r in self.regions if not r.cues) > 1:
            raise ValueError(f"{self.qid}: at most one region may have no cue")
        for r in self.regions:
            if len(r.dist) != self.width:
                raise ValueError(
                    f"{self.qid}/{r.name}: dist has {len(r.dist)} entries, not {self.width}"
                )
            if min(r.dist) < 0 or not math.isclose(sum(r.dist), 1.0, abs_tol=1e-9):
                raise ValueError(f"{self.qid}/{r.name}: dist is not a distribution")
            certain = max(r.dist) == 1.0
            if certain != (r.kind == "deterministic"):
                raise ValueError(f"{self.qid}/{r.name}: kind {r.kind!r} does not match its dist")
        cues = [(r.name, c) for r in self.regions for c in r.cues]
        for i, (ra, a) in enumerate(cues):
            for rb, b in cues[i + 1 :]:
                same = (a in b or b in a) if isinstance(a, str) and isinstance(b, str) else a == b
                if same:
                    raise ValueError(f"{self.qid}: cues of {ra} and {rb} cannot be told apart")


@dataclass(frozen=True)
class Domain:
    name: str
    questions: tuple[QuestionSpec, ...]
    render: Callable[[random.Random, dict[str, Cue]], Text]  # planted cues by qid -> state
    filler: tuple[str, ...] = ()  # everything else render may write, for the collision check
    hide: tuple[str, ...] = ()  # questions whose cues are withheld in an unrecoverable corpus

    def check(self) -> None:
        """Raise unless every question is well formed and no cue can leak into another question.

        A department keyword that was also an urgency cue would tie two
        questions' regions together, and the recorded truth for both would be
        wrong. So the surface forms of different questions, and of the filler,
        must not contain one another.
        """
        for q in self.questions:
            q.check()
        if not set(self.hide) <= {q.qid for q in self.questions}:
            raise ValueError(f"{self.name}: hide names a question that does not exist")
        owned = [
            (q.qid, s)
            for q in self.questions
            for r in q.regions
            for c in r.cues
            for s in surfaces(c)
        ]
        owned += [("filler", s) for s in self.filler]
        for i, (qa, a) in enumerate(owned):
            for qb, b in owned[i + 1 :]:
                if qa != qb and (a in b or b in a):
                    raise ValueError(f"{self.name}: {qa} and {qb} share the surface {a!r} / {b!r}")
