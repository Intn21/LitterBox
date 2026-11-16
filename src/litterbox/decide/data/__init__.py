"""Training and evaluation data for decision models.

One format for everything, :class:`~litterbox.decide.data.record.Record`, and a
generator whose label frequencies are written down, so that "honest" has a
known answer. See ``docs/decision-model.md``, step 1.
"""

from litterbox.decide.data.domains import DOMAINS
from litterbox.decide.data.generate import generate
from litterbox.decide.data.record import Record, read_jsonl, write_jsonl
from litterbox.decide.data.regions import Domain, QuestionSpec, Region

__all__ = [
    "DOMAINS",
    "Domain",
    "QuestionSpec",
    "Record",
    "Region",
    "generate",
    "read_jsonl",
    "write_jsonl",
]
