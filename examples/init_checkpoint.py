"""Save a checkpoint of a freshly built model: random weights, step 0.

Run it::

    python examples/init_checkpoint.py            # -> runs/fresh-tinystories-dense
    python examples/init_checkpoint.py configs/models/tinystories-swa-hybrid.yaml
    python examples/init_checkpoint.py --out runs/my-init --seed 7

Then anything that reads a run directory works on it::

    python examples/generate.py runs/fresh-tinystories-dense --prompt "Once upon a time"

The output is nonsense, which is the point: it is the before picture. The same
checkpoint is also a fixed starting point for training — copy it to a run's
directory as ``latest.pt`` and ``train_tinystories.py`` resumes from step 0 with
exactly these weights, so two runs can differ in nothing but their config.

The checkpoint records the model config that built it, so it needs no matching
file under ``configs/models/`` to be loaded later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from litterbox.model import build_model
from litterbox.utils.config import load_model_config

DEFAULT = "configs/models/tinystories-dense.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", nargs="?", default=DEFAULT, help="a model config")
    parser.add_argument("--out", help="run directory to write; default runs/fresh-<config name>")
    parser.add_argument("--seed", type=int, default=0, help="seeds the random weights")
    parser.add_argument("--force", action="store_true", help="overwrite an existing checkpoint")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    out = Path(args.out) if args.out else Path("runs") / f"fresh-{Path(args.config).stem}"
    ckpt = out / "latest.pt"
    if ckpt.exists() and not args.force:
        raise SystemExit(
            f"{ckpt} already exists. It may hold trained weights; pass --force to overwrite, "
            f"or --out to write somewhere else."
        )

    torch.manual_seed(args.seed)
    model = build_model(cfg)
    out.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": None,  # nothing has trained, so there is no optimizer state
            "step": 0,
            "config": None,
            "model_config": cfg.model_dump(),
            "seed": args.seed,
        },
        tmp,
    )
    tmp.replace(ckpt)

    n = sum(p.numel() for p in model.parameters())
    pattern = [layer.mixer for layer in cfg.layer_pattern]
    print(
        f"wrote {ckpt}: {n / 1e6:.2f}M random parameters, seed {args.seed}, "
        f"{pattern} x{cfg.n_layers // len(pattern)}, step 0"
    )
    print(f'\n    python examples/generate.py {out} --prompt "Once upon a time"')


if __name__ == "__main__":
    main()
