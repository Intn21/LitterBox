"""Write a generated corpus to disk: three JSONL splits and a manifest.

    litterbox-decide-data --domain tickets --n 20000 --seed 0 --out data/decide/tickets
    litterbox-decide-data --domain tickets --n 5000 --seed 1 --unrecoverable \\
        --out data/decide/tickets-unrecoverable

The split is 80/10/10 into ``train``, ``calibration`` and ``test``. The
calibration slice is where temperatures are fitted; the test slice is where a
model's numbers are held against the recorded truth. ``manifest.json`` records
everything needed to regenerate the corpus and to read it: the seed, every
region's die and weight, and how many records fell in each.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from litterbox.decide.data.domains import DOMAINS
from litterbox.decide.data.generate import generate
from litterbox.decide.data.record import write_jsonl

SPLITS = (("train", 0.8), ("calibration", 0.1), ("test", 0.1))


def _commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def build(domain_name: str, n: int, seed: int, unrecoverable: bool, out: str | Path) -> dict:
    """Generate, split, write; returns the manifest."""
    domain = DOMAINS[domain_name]
    records = generate(domain, n, seed, unrecoverable=unrecoverable)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sizes, start = {}, 0
    for i, (name, share) in enumerate(SPLITS):
        end = n if i == len(SPLITS) - 1 else start + round(n * share)
        sizes[name] = write_jsonl(out / f"{name}.jsonl", records[start:end])
        start = end
    manifest = {
        "domain": domain.name,
        "source": records[0].source if records else None,
        "n": n,
        "seed": seed,
        "unrecoverable": unrecoverable,
        "hidden_questions": list(domain.hide) if unrecoverable else [],
        "splits": sizes,
        "commit": _commit(),
        "questions": {
            q.qid: {
                "type": q.type,
                "options": list(q.options) if q.type != "noul" else ["no", "yes"],
                "base_rate": [round(v, 6) for v in q.base_rate()],
                "regions": [
                    {
                        "name": r.name,
                        "kind": r.kind,
                        "dist": list(r.dist),
                        "weight": r.weight,
                        "cues": len(r.cues),
                    }
                    for r in q.regions
                ],
                "records_per_region": dict(
                    sorted(Counter(r.region[q.qid] for r in records).items())
                ),
            }
            for q in domain.questions
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a decision-model corpus with known label frequencies."
    )
    ap.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    ap.add_argument("--n", type=int, default=20000, help="records to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--unrecoverable", action="store_true", help="withhold the domain's hidden cues"
    )
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    manifest = build(args.domain, args.n, args.seed, args.unrecoverable, args.out)
    print(f"{manifest['source']}: {manifest['splits']} -> {args.out}")
    for qid, q in manifest["questions"].items():
        print(f"  {qid:<16} {q['records_per_region']}")


if __name__ == "__main__":
    main()
