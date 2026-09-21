"""Train a small model on TinyStories and watch it learn to write.

Run it::

    python examples/train_tinystories.py
    python examples/train_tinystories.py training.max_steps=300 logging.out_dir=runs/smoke
    python examples/train_tinystories.py --d-model 384 --layers 8 training.compile=true   # on CUDA

Anything after the flags is an OmegaConf override applied to the training
config. The first run packs TinyStories (~90 s, ~1 GB under ``data/``); later
runs find it up to date and start immediately. Interrupt and re-run to resume
from the last checkpoint.

The model is assembled by hand from arguments rather than from a model config,
which is ROADMAP work still to come. The default shape — 256 wide, 6 layers —
is 17.6M parameters, of which 12.9M are the embedding table: the GPT-2
vocabulary has 50,257 entries and children's stories use a few thousand.

NOTE the ``if __name__ == "__main__"`` guard. Packing may use worker processes,
and on macOS each worker re-imports this module.
"""

from __future__ import annotations

import argparse

import torch

from litterbox.infer import generate_uncached
from litterbox.model import dense_transformer
from litterbox.train import load_run_config, train

PROMPT = "Once upon a time"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="configs/training/tinystories-small.yaml")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=50_304, help="50,257 padded to 64s")
    parser.add_argument("--mixer", default="full_attention")
    parser.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    parser.add_argument("overrides", nargs="*", help="e.g. training.max_steps=300")
    args = parser.parse_args()

    cfg = load_run_config(args.config, args.overrides)
    torch.manual_seed(cfg.training.seed)
    model = dense_transformer(
        args.vocab_size,
        args.d_model,
        args.layers,
        args.heads,
        args.kv_heads,
        max_seq_len=cfg.training.seq_len,
        mixer=args.mixer,
    )

    # The tokenizer is only known once the data is prepared, so resolve it lazily
    # inside the callback rather than up front.
    from litterbox.data import load_data_config
    from litterbox.data.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(load_data_config(cfg.data.config).tokenizer)

    def show_sample(step: int, model: torch.nn.Module) -> None:
        device = next(model.parameters()).device
        prompt = torch.tensor([tokenizer.encode(PROMPT)], device=device)
        out = generate_uncached(
            model,
            prompt,
            max_new_tokens=80,
            max_context=cfg.training.seq_len,
            temperature=0.8,
            top_k=40,
            eos_id=tokenizer.eos_id,
            generator=torch.Generator().manual_seed(step),
        )
        text = tokenizer.decode(out[0].tolist()).replace("\n", " ")
        print(f"         sample: {text}\n")

    result = train(model, cfg, tokenizer=tokenizer, resume=not args.fresh, on_eval=show_sample)
    print(
        f"done: step {result.step:,} on {result.device}, "
        f"train loss {result.train_loss:.3f}, val loss {result.val_loss:.3f}"
    )


if __name__ == "__main__":
    main()
