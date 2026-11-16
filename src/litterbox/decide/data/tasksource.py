"""Public decision data, through one adapter: tasksource-jev-typed-decisions.

Tasksource has already recast about 750 public classification, rating and
multiple-choice datasets as typed decisions, with the source's own labels, a
license class per row, and a group id tying together rows that share a state.
A survey on 2026-11-16 found nothing with cleaner provenance at that scale, so
instead of converting datasets one at a time this module maps tasksource rows
onto :class:`~litterbox.decide.data.record.Record` and stops there.

One row is one question about one state:

==========  ======================================  ==============================
kind        tasksource                              record
==========  ======================================  ==============================
``choice``  ``options`` names, ``target`` vector    choice, options as criteria
``score``   ``options`` levels, ``target`` vector   score; choice if over 10 levels
``noul``    no options, ``target = [p_yes]``        noul, answers ordered no, yes
==========  ======================================  ==============================

A one-hot target becomes a label and no truth. A soft target, annotator vote
shares or a mean rating, becomes a truth, with the label its most likely
answer; that 13% of the data is the nearest thing public sources have to the
generator's answer key.

Two filters are applied here because getting them wrong is silent. Sources
that also feed an evaluation set are dropped (:data:`JEV_BENCH_SOURCES`), so
that those benchmarks stay zero-shot. And a calibration slice is carved out of
train by hashing the group id, never the row, because a quarter of the rows
share a state with another row and a row-level split would leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from litterbox.decide.data.record import Record
from litterbox.decide.schema import MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS

DATASET = "tasksource/tasksource-jev-typed-decisions"

# The 22 sources of Praveenrajus/jev-bench, as patterns over tasksource's source names.
# Training on none of them keeps that benchmark a test of sources the model has not seen.
JEV_BENCH_SOURCES = (
    r"banking77",
    r"clinc",
    r"massive",
    r"ledgar",
    r"go_?emotions",
    r"mmlu",
    r"(^|[/_-])(ai2_)?arc($|[/_-])",
    r"mnli",
    r"(^|[/_-])sst",
    r"yelp",
    r"helpsteer2",
    r"sts-?b",
    r"measuring.hate",
    r"boolq",
    r"fever",
    r"paws",
    r"civil_comments",
    r"sms_spam",
    r"strategy.?qa",
)


class Skip(Exception):
    """A row that cannot or should not become a record; the message is the reason counted."""


def to_record(row: dict) -> Record:
    """One tasksource row as a single-question record. Raises :class:`Skip` with the reason."""
    kind, options, target = row["kind"], list(row["options"]), [float(t) for t in row["target"]]
    qid = row.get("question_id") or "decision"
    if kind == "noul":
        if len(target) != 1 or not 0.0 <= target[0] <= 1.0:
            raise Skip("bad_target")
        dist = [1.0 - target[0], target[0]]  # our nouls are ordered no, yes
        question = {"type": "noul", "instructions": row["question"]}
    else:
        total = sum(target)
        if len(target) != len(options) or min(target, default=-1) < 0 or abs(total - 1) > 1e-3:
            raise Skip("bad_target")
        dist = [t / total for t in target]
        if len(set(options)) != len(options) or not all(options):
            raise Skip("duplicate_or_empty_options")
        if kind == "score" and len(options) <= MAX_SCORE_LEVELS:
            question = {"type": "score", "instructions": row["question"], "criteria": options}
        elif len(options) <= MAX_CHOICE_OPTIONS:  # a choice, or a scale too long to be a score
            criteria = {o: "" for o in options}
            question = {"type": "choice", "instructions": row["question"], "criteria": criteria}
        else:
            raise Skip("too_many_options")
    soft = max(dist) < 1.0 - 1e-6
    try:
        return Record(
            id=row["id"],
            source=f"tasksource/{row['source']}",
            state=row["state"],
            questions={qid: question},
            labels={qid: max(range(len(dist)), key=dist.__getitem__)},  # first index on ties
            truth={qid: dist} if soft else None,
            group=row.get("group_id") or row["id"],
            license=row.get("license_use"),
        )
    except ValidationError as e:
        raise Skip("invalid") from e


def convert(
    rows: Iterable[dict],
    *,
    exclude: Iterable[str] = JEV_BENCH_SOURCES,
    licenses: Iterable[str] | None = None,
    stats: Counter | None = None,
) -> Iterator[Record]:
    """Records for the rows that pass the filters; ``stats`` counts what happened to the rest.

    ``exclude`` holds case-insensitive patterns over the source name.
    ``licenses`` keeps only those ``license_use`` classes (``commercial``,
    ``non-commercial``, ``unspecified``); ``None`` keeps all, which is right
    for research and wrong for anything shipped.
    """
    pattern = re.compile("|".join(f"(?:{p})" for p in exclude), re.IGNORECASE) if exclude else None
    allowed = None if licenses is None else set(licenses)
    stats = Counter() if stats is None else stats
    for row in rows:
        if pattern is not None and pattern.search(row["source"]):
            stats["skipped/excluded_source"] += 1
            stats[f"excluded/{row['source']}"] += 1
            continue
        if allowed is not None and row.get("license_use") not in allowed:
            stats["skipped/license"] += 1
            continue
        try:
            record = to_record(row)
        except Skip as reason:
            stats[f"skipped/{reason}"] += 1
            continue
        stats["kept"] += 1
        yield record


def in_slice(group: str, share: float, seed: int = 0) -> bool:
    """Whether a group falls in a slice of the given share. Stable across runs and machines."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


def _summary(records: list[Record]) -> dict:
    kinds = Counter(next(iter(r.questions.values())).type for r in records)
    return {
        "records": len(records),
        "groups": len({r.group for r in records}),
        "kinds": dict(kinds),
        "soft_targets": sum(r.truth is not None for r in records),
        "licenses": dict(Counter(r.license for r in records)),
        "sources": dict(Counter(r.source for r in records).most_common()),
    }


def build(
    out: str | Path,
    *,
    config: str = "default",
    limit: int | None = 200_000,
    calibration_share: float = 0.05,
    exclude: Iterable[str] = JEV_BENCH_SOURCES,
    licenses: Iterable[str] | None = None,
    seed: int = 0,
    revision: str | None = None,
    rows_by_split: dict[str, Iterable[dict]] | None = None,
) -> dict:
    """Write ``train``, ``calibration``, ``validation`` and ``test`` JSONL plus a manifest.

    Train is a shuffled sample of ``limit`` rows of tasksource's train split,
    with ``calibration_share`` of its groups moved to the calibration file.
    Validation and test are tasksource's own, whole. ``rows_by_split`` supplies
    the rows directly, for tests; otherwise they are streamed from the Hub.
    """
    from litterbox.decide.data.record import write_jsonl

    if rows_by_split is None:
        from datasets import load_dataset  # the 'eval' extra

        def stream(split):
            ds = load_dataset(DATASET, config, split=split, streaming=True, revision=revision)
            if split == "train":
                ds = ds.shuffle(seed=seed, buffer_size=50_000)
                if limit is not None:
                    ds = ds.take(limit)
            return ds

        rows_by_split = {s: stream(s) for s in ("train", "validation", "test")}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = tuple(exclude)
    stats: Counter = Counter()
    pools = {
        split: list(convert(rows, exclude=exclude, licenses=licenses, stats=stats))
        for split, rows in rows_by_split.items()
    }
    train = pools.pop("train", [])
    pools = {
        "train": [r for r in train if not in_slice(r.group, calibration_share, seed)],
        "calibration": [r for r in train if in_slice(r.group, calibration_share, seed)],
        **pools,
    }
    for split, records in pools.items():
        write_jsonl(out / f"{split}.jsonl", records)

    groups = {split: {r.group for r in records} for split, records in pools.items()}
    names = list(pools)
    manifest = {
        "dataset": DATASET,
        "config": config,
        "revision": revision,
        "seed": seed,
        "train_limit": limit,
        "calibration_share": calibration_share,
        "license_filter": None if licenses is None else sorted(licenses),
        "excluded_patterns": list(exclude),
        "excluded_sources": {
            k.split("/", 1)[1]: v for k, v in sorted(stats.items()) if k.startswith("excluded/")
        },
        "skipped": {k.split("/", 1)[1]: v for k, v in stats.items() if k.startswith("skipped/")},
        "shared_groups": {
            f"{a}&{b}": len(groups[a] & groups[b])
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "splits": {split: _summary(records) for split, records in pools.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adapt tasksource typed decisions to litterbox records."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="default")
    ap.add_argument("--limit", type=int, default=200_000, help="train rows to sample; 0 for all")
    ap.add_argument("--calibration-share", type=float, default=0.05)
    ap.add_argument("--licenses", nargs="*", help="keep only these license_use classes")
    ap.add_argument("--no-exclude", action="store_true", help="keep the jev-bench sources too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(DATASET).sha  # pin what we read, and record it
    manifest = build(
        args.out,
        config=args.config,
        limit=args.limit or None,
        calibration_share=args.calibration_share,
        exclude=() if args.no_exclude else JEV_BENCH_SOURCES,
        licenses=args.licenses,
        seed=args.seed,
        revision=revision,
    )
    print(f"{DATASET}@{revision[:8]} -> {args.out}")
    for split, s in manifest["splits"].items():
        print(f"  {split:<12} {s['records']:>8,} records  {s['groups']:>8,} groups  {s['kinds']}")
    print(f"  skipped {manifest['skipped']}")
    print(f"  groups shared between splits {manifest['shared_groups']}")


if __name__ == "__main__":
    main()
