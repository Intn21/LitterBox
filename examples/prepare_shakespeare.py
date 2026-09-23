"""Fetch Tiny Shakespeare and split it for training.

Run it::

    python examples/prepare_shakespeare.py
    litterbox-pack configs/data/tinyshakespeare.yaml

Tiny Shakespeare is one 1.1 MB text file: every play, concatenated, 65 distinct
characters. It is the corpus nanoGPT's character-level example trains on, and
it is small enough that a laptop finishes a run in minutes.

One file is one document, and the repo's holdout splits by *document*, so it
cannot carve a validation set out of this on its own. Do what nanoGPT does
instead: the first 90% of the characters are training text, the last 10% are
validation. The split is by position, which is fine here — the file is one
continuous stream and a play does not repeat itself the way web text does.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
RAW = Path("data/raw/tinyshakespeare")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    source = RAW / "input.txt"
    if not source.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, source)
    text = source.read_text(encoding="utf-8")
    cut = int(0.9 * len(text))
    (RAW / "train.txt").write_text(text[:cut], encoding="utf-8")
    (RAW / "val.txt").write_text(text[cut:], encoding="utf-8")
    print(
        f"{len(text):,} characters, {len(set(text))} distinct -> "
        f"train {cut:,} / val {len(text) - cut:,} under {RAW}/"
    )
    print("\n    litterbox-pack configs/data/tinyshakespeare.yaml")


if __name__ == "__main__":
    main()
