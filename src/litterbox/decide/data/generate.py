"""The engine: a domain's regions in, records with their truth out.

One record is made in five moves, all from one seeded generator:

1. for each question, pick a region by weight;
2. draw the label from that region's die;
3. choose one surface form of each picked region's cue;
4. ask the domain to render a state from those cues plus filler;
5. shuffle each choice's options, and write the label and the truth in the
   shuffled order.

With ``unrecoverable=True`` the questions a domain lists in ``hide`` skip move
3: the region is still rolled and the label still drawn from it, but nothing
about it reaches the state. The truth recorded for such a question is the base
rate, the regions mixed by weight, because that is the most anyone reading
the state can know.
"""

from __future__ import annotations

import random

from litterbox.decide.data.record import Record
from litterbox.decide.data.regions import Cue, Domain


def generate(domain: Domain, n: int, seed: int = 0, *, unrecoverable: bool = False) -> list[Record]:
    """``n`` records of ``domain``; the same arguments always give the same records."""
    domain.check()
    rng = random.Random(seed)
    hidden = set(domain.hide) if unrecoverable else set()
    source = f"generated/{domain.name}" + ("-unrecoverable" if unrecoverable else "")
    records = []
    for i in range(n):
        planted: dict[str, Cue] = {}
        questions, labels, truth, region_of = {}, {}, {}, {}
        for q in domain.questions:
            region = rng.choices(q.regions, weights=[r.weight for r in q.regions])[0]
            answer = rng.choices(range(q.width), weights=region.dist)[0]  # canonical index
            if q.qid in hidden:
                dist, region_of[q.qid] = q.base_rate(), "unrecoverable/hidden"
            else:
                dist, region_of[q.qid] = list(region.dist), f"{region.kind}/{region.name}"
                if region.cues:
                    planted[q.qid] = rng.choice(region.cues)
            order = list(range(q.width))
            if q.type == "choice":  # a score's levels are ordered; a noul is always no, yes
                rng.shuffle(order)
            questions[q.qid] = q.as_question(rng.choice(q.instructions), order)
            labels[q.qid] = order.index(answer)
            truth[q.qid] = [dist[j] for j in order]
        records.append(
            Record(
                id=f"{domain.name}-{seed}-{i:06d}",
                source=source,
                state=domain.render(rng, planted),
                questions=questions,
                labels=labels,
                truth=truth,
                region=region_of,
            )
        )
    return records
