"""Train a small model on TinyStories and watch it learn to write.

Run it::

    python examples/train_tinystories.py
    python examples/train_tinystories.py --model configs/models/tinystories-swa-hybrid.yaml
    python examples/train_tinystories.py training.max_steps=300 logging.out_dir=runs/smoke
    python examples/train_tinystories.py model.tier=fast training.compile=true      # on CUDA

The model comes from ``--model``, a YAML under ``configs/models/``, and the
training recipe and corpus from ``--config``. The name is historical: with
``--config configs/training/shakespeare-char.yaml`` it trains on Tiny
Shakespeare at the character level instead. Swapping the
token mixer — full attention, sliding window, a hybrid of the two — is choosing a
different file; nothing in this script, the loop, or generation changes.

Anything after the flags is an OmegaConf override: ``training.*`` and
``logging.*`` go to the training config, ``model.*`` to the model config. Each
model logs to ``runs/<model name>`` unless told otherwise.

The first run packs TinyStories (~90 s, ~1 GB under ``data/``); later runs find
it up to date and start immediately. Interrupt and re-run to resume from the
last checkpoint.

NOTE the ``if __name__ == "__main__"`` guard. Packing may use worker processes,
and on macOS each worker re-imports this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from litterbox.infer import generate
from litterbox.model import build_model
from litterbox.train import load_run_config, train
from litterbox.utils.config import load_model_config

DEFAULT_PROMPT = "Once upon a time"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/training/tinystories-small.yaml")
    parser.add_argument("--model", default="configs/models/tinystories-dense.yaml")
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="what each sample starts from")
    parser.add_argument("overrides", nargs="*", help="e.g. training.max_steps=300 model.tier=fast")
    args = parser.parse_args()

    model_overrides = [o[len("model.") :] for o in args.overrides if o.startswith("model.")]
    run_overrides = [o for o in args.overrides if not o.startswith("model.")]
    if not any(o.startswith("logging.out_dir=") for o in run_overrides):
        run_overrides.append(f"logging.out_dir=runs/{Path(args.model).stem}")

    cfg = load_run_config(args.config, run_overrides)
    model_cfg = load_model_config(args.model, model_overrides)
    torch.manual_seed(cfg.training.seed)
    model = build_model(model_cfg)
    print(
        f"{Path(args.model).stem}: {[layer.mixer for layer in model_cfg.layer_pattern]} "
        f"x{model_cfg.n_layers // len(model_cfg.layer_pattern)}"
    )

    # The tokenizer is only known once the data is prepared, so resolve it lazily
    # inside the callback rather than up front.
    from litterbox.data import load_data_config
    from litterbox.data.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(load_data_config(cfg.data.config).tokenizer)

    def show_sample(step: int, model: torch.nn.Module) -> None:
        device = next(model.parameters()).device
        prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        out = generate(
            model,
            prompt,
            max_new_tokens=80,
            max_context=cfg.training.seq_len,
            temperature=0.8,
            top_k=40,
            eos_id=tokenizer.eos_id,
            generator=torch.Generator().manual_seed(step),
        )
        text = tokenizer.decode(out[0].tolist()).replace("\n", " | ")
        print(f"         sample: {text}\n")

    result = train(model, cfg, tokenizer=tokenizer, resume=not args.fresh, on_eval=show_sample)
    print(
        f"done: step {result.step:,} on {result.device}, "
        f"train loss {result.train_loss:.3f}, val loss {result.val_loss:.3f}"
    )


if __name__ == "__main__":
    main()
