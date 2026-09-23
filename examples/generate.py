"""Generate text from a trained checkpoint.

Run it::

    python examples/generate.py runs/tinystories-dense
    python examples/generate.py runs/tinystories-swa-hybrid --prompt "Once upon a time, a dragon"
    python examples/generate.py runs/tinystories-dense --greedy --tokens 200
    python examples/generate.py runs/tinystories-dense --seed 1 --n 3

The first argument is a run directory (containing ``latest.pt``) or a checkpoint
file. The model's shape comes from a model config; by default that is
``configs/models/<run name>.yaml``, and ``--model`` overrides it.

Generation goes through the KV cache. ``--check`` also runs the cache-free
oracle and confirms the two agree token for token.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from litterbox.data.tokenizer import build_tokenizer
from litterbox.infer import generate, generate_uncached
from litterbox.model import build_model
from litterbox.train import load_checkpoint, pick_device
from litterbox.utils.config import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="a run directory under runs/, or a .pt checkpoint")
    parser.add_argument("--model", help="model config; default configs/models/<run name>.yaml")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=150, help="new tokens to generate")
    parser.add_argument("--n", type=int, default=1, help="number of samples")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true", help="always take the likeliest token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="verify against cache-free generation")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run = Path(args.run)
    ckpt = run if run.suffix == ".pt" else run / "latest.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    model_cfg_path = Path(args.model) if args.model else Path("configs/models") / f"{run.stem}.yaml"
    if not model_cfg_path.exists():
        raise SystemExit(
            f"no model config at {model_cfg_path}; pass --model with the config this run was "
            f"trained with"
        )

    cfg = load_model_config(model_cfg_path)
    model = build_model(cfg)
    step = load_checkpoint(ckpt, model)
    device = pick_device(args.device)
    model.to(device).eval()

    tokenizer = build_tokenizer({"type": "tiktoken", "encoding": "gpt2"})
    prompt = torch.tensor([tokenizer.encode(args.prompt)] * args.n, device=device)
    budget = cfg.max_seq_len - prompt.shape[1]
    tokens = min(args.tokens, budget)
    if tokens < args.tokens:
        print(f"(capped at {tokens} new tokens: the context is {cfg.max_seq_len})")

    pattern = [layer.mixer for layer in cfg.layer_pattern]
    print(
        f"{run.stem}: step {step:,}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
        f"parameters, {pattern} x{cfg.n_layers // len(pattern)}, on {device}\n"
    )

    kwargs = dict(
        max_context=cfg.max_seq_len,
        temperature=0.0 if args.greedy else args.temperature,
        top_k=None if args.greedy else args.top_k,
        eos_id=tokenizer.eos_id,
    )
    t0 = time.perf_counter()
    out = generate(
        model, prompt, tokens, generator=torch.Generator().manual_seed(args.seed), **kwargs
    )
    elapsed = time.perf_counter() - t0

    for i, ids in enumerate(out):
        text = tokenizer.decode(ids.tolist())
        print(f"--- sample {i + 1} ---" if args.n > 1 else "---")
        print(text.strip(), "\n")

    new = out.shape[1] - prompt.shape[1]
    print(f"{new * args.n} tokens in {elapsed:.2f}s ({new * args.n / elapsed:.0f} tok/s)")

    if args.check:
        oracle = generate_uncached(
            model, prompt, tokens, generator=torch.Generator().manual_seed(args.seed), **kwargs
        )
        print("cached == cache-free:", torch.equal(out, oracle))


if __name__ == "__main__":
    main()
