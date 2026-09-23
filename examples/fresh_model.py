"""The whole stack on a fresh model: no data, no checkpoint, no training.

Run it::

    python examples/fresh_model.py
    python examples/fresh_model.py --model configs/models/tinystories-swa-hybrid.yaml
    python examples/fresh_model.py --prompt "Once upon a time" --tokens 40

Builds a model from a YAML config, shows what a forward pass costs and returns,
checks the one property an untrained model must have, generates from a prompt
(it will be nonsense — that is the point), confirms the KV cache reproduces
cache-free generation exactly, and swaps the token mixer to show the shape of
the inference state change with it. Runs in a few seconds on a CPU.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from litterbox.infer import generate, generate_uncached
from litterbox.model import build_model
from litterbox.train import pick_device
from litterbox.utils.config import load_model_config

CONFIGS = {
    "dense": "configs/models/tinystories-dense.yaml",
    "swa": "configs/models/tinystories-swa.yaml",
    "hybrid": "configs/models/tinystories-swa-hybrid.yaml",
}


def tokenizer_or_bytes():
    """GPT-2's tokenizer if tiktoken is installed and cached; the byte tokenizer otherwise."""
    from litterbox.data.tokenizer import build_tokenizer

    try:
        return build_tokenizer({"type": "tiktoken", "encoding": "gpt2"}), 50_304
    except Exception:  # not installed, or no cached vocabulary and no network
        print("(tiktoken unavailable: using the byte-level tokenizer, vocab 256)\n")
        return build_tokenizer({"type": "byte"}), 256


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=CONFIGS["dense"])
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tokenizer, vocab = tokenizer_or_bytes()

    # ---------------------------------------------------------------- build
    section(f"1. Build a model from {args.model}")
    cfg = load_model_config(args.model, [f"vocab_size={vocab}"])
    model = build_model(cfg).to(device).eval()
    pattern = [layer.mixer for layer in cfg.layer_pattern]
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.tok_emb.weight.numel()
    print(f"layer_pattern {pattern} x{cfg.n_layers // len(pattern)} -> {cfg.n_layers} blocks")
    print(f"d_model {cfg.d_model}, max_seq_len {cfg.max_seq_len}, vocab {cfg.vocab_size:,}")
    print(
        f"{n_params / 1e6:.2f}M parameters, of which {n_embed / 1e6:.2f}M "
        f"({n_embed / n_params:.0%}) are the embedding table, on {device}"
    )
    for i, block in enumerate(model.blocks):
        m = block.mixer
        extra = f", window {m.window}" if hasattr(m, "window") else ""
        print(f"   block {i}: {type(m).__name__}, {m.heads} heads / {m.kv_heads} kv heads{extra}")

    # -------------------------------------------------------------- forward
    section("2. A forward pass, and the one thing an untrained model must get right")
    ids = torch.randint(0, cfg.vocab_size, (4, 65), device=device)
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        elapsed = time.perf_counter() - t0
    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()).item()
    print(f"ids {tuple(x.shape)} -> logits {tuple(logits.shape)} in {elapsed * 1e3:.0f} ms")
    print(f"loss on random tokens: {loss:.3f}    ln(vocab_size) = {math.log(cfg.vocab_size):.3f}")
    print(
        "An untrained model should be maximally unsure, and ln(vocab) is exactly that. Anything\n"
        "meaningfully lower means the targets are leaking into the inputs somewhere."
    )

    # ------------------------------------------------------------- generate
    section(f"3. Generate from {args.prompt!r}")
    prompt = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    gen = torch.Generator().manual_seed(args.seed)
    out = generate(
        model,
        prompt,
        args.tokens,
        max_context=cfg.max_seq_len,
        temperature=0.8,
        top_k=40,
        generator=gen,
    )
    print(repr(tokenizer.decode(out[0].tolist())))
    print(
        "\nNonsense, as it should be: every weight is random, so every next token is a coin flip\n"
        "over the vocabulary. Training is what turns this into stories."
    )

    # ---------------------------------------------------------------- cache
    section("4. The KV cache reproduces cache-free generation exactly, trained or not")
    greedy = dict(max_context=cfg.max_seq_len, temperature=0.0)
    a = generate(model, prompt, args.tokens, **greedy)
    b = generate_uncached(model, prompt, args.tokens, **greedy)
    print(f"cached == cache-free, greedy, {args.tokens} tokens: {torch.equal(a, b)}")
    states = model.init_states(1, cfg.max_seq_len)
    with torch.no_grad():
        model(prompt, states, pos_offset=0)
    print("per-block inference state after the prompt:")
    for i, state in enumerate(states):
        print(f"   block {i}: {state.kv}")

    # ----------------------------------------------------------------- swap
    section("5. Swap the token mixer: the same script, a different file")
    for name, path in CONFIGS.items():
        other = build_model(load_model_config(path, [f"vocab_size={vocab}"])).to(device).eval()
        states = other.init_states(1, cfg.max_seq_len)
        with torch.no_grad():
            other(torch.randint(0, vocab, (1, cfg.max_seq_len), device=device), states)
        kb = [round(s.kv.nbytes_used / 1e3) for s in states]
        n = sum(p.numel() for p in other.parameters())
        print(
            f"   {name:7} {n / 1e6:.2f}M params   cache after {cfg.max_seq_len} tokens, "
            f"per block (kB): {kb}   total {sum(kb)} kB"
        )
    print(
        "\nSame parameter count, same code path; only the caches differ. That is what the configs\n"
        "change, and nothing in this script knows which one it was handed."
    )


if __name__ == "__main__":
    main()
