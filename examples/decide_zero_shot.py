"""Zero-shot decisions from a pretrained chat model: the step 0 baseline.

Run it on a pod::

    python examples/decide_zero_shot.py --dataset boolq --limit 2000 \\
        --out experiments/2026-11-09-zero-shot-qwen3.5-9b-boolq

It loads a Hugging Face causal LM, turns each dataset row into a decision
request, answers it with the baseline lettered-prompt readout
(:mod:`litterbox.decide.readout`), and writes the calibration summary
(:mod:`litterbox.decide.calibration`) plus every per-item probability to the
experiment folder. Nothing is trained. The number this produces is the one
every later step has to beat.

Datasets:

- ``boolq``: a passage and a yes/no question, as a ``noul``. 3,270 validation items.
- ``banking77``: a customer message and 77 intents, as a ``choice``. The baseline
  readout has 62 single-token labels, so ``--intents N`` keeps the first N
  intents and the rows that belong to them; the pointer head lifts that limit.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from litterbox.data.tokenizer.external import HFTokenizer
from litterbox.decide.calibration import reliability, summarize
from litterbox.decide.readout import ALPHABET, batched_probabilities, chat_wrap, hf_logits, render
from litterbox.decide.schema import Request

BOOLQ_INSTRUCTIONS = "Based on the passage, is the answer to this question yes? {question}"


def boolq(limit: int | None):
    from datasets import load_dataset

    rows = load_dataset("google/boolq", split="validation")
    if limit:
        rows = rows.select(range(min(limit, len(rows))))
    requests, labels = [], []
    for r in rows:
        req = Request(
            model="baseline",
            state=r["passage"],
            questions={
                "yes": {
                    "type": "noul",
                    "instructions": BOOLQ_INSTRUCTIONS.format(question=r["question"]),
                }
            },
        )
        requests.append(("yes", req))
        labels.append(1 if r["answer"] else 0)  # noul order is (no, yes)
    return requests, labels, {"name": "google/boolq", "split": "validation"}


def banking77(limit: int | None, intents: int):
    from datasets import load_dataset

    if intents > len(ALPHABET):
        raise SystemExit(f"--intents {intents} exceeds the {len(ALPHABET)} labels of this readout")
    rows = load_dataset("PolyAI/banking77", split="test")
    names = rows.features["label"].names[:intents]
    criteria = {name.replace("_", " "): "" for name in names}
    rows = rows.filter(lambda r: r["label"] < intents)
    if limit:
        rows = rows.select(range(min(limit, len(rows))))
    requests, labels = [], []
    for r in rows:
        req = Request(
            model="baseline",
            state=r["text"],
            questions={
                "intent": {
                    "type": "choice",
                    "instructions": "What is the customer's intent?",
                    "criteria": criteria,
                }
            },
        )
        requests.append(("intent", req))
        labels.append(r["label"])
    return requests, labels, {"name": "PolyAI/banking77", "split": "test", "intents": intents}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--dataset", choices=["boolq", "banking77"], default="boolq")
    ap.add_argument("--limit", type=int, default=None, help="rows to use; default all")
    ap.add_argument("--intents", type=int, default=20, help="banking77 only: first N intents")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-chat-template", action="store_true", help="bare prompts, for base models")
    ap.add_argument("--out", required=True, help="experiment folder; results.json goes here")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  model {args.model}  dataset {args.dataset}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    hf_tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    print(f"loaded in {time.time() - t0:.0f}s")
    tokenizer = HFTokenizer(args.model)

    requests, labels, meta = (
        boolq(args.limit) if args.dataset == "boolq" else banking77(args.limit, args.intents)
    )
    wrap = None if args.no_chat_template else chat_wrap(hf_tok, enable_thinking=False)
    prompts = [
        render(req, qid, answer_line=wrap is None)
        if wrap is None
        else wrap(render(req, qid, answer_line=False))
        for qid, req in requests
    ]
    widths = [2 if qid == "yes" else len(req.questions[qid].criteria) for qid, req in requests]
    print(f"{len(prompts)} prompts; first one:\n---\n{prompts[0]}\n---")

    t0 = time.time()
    probs, tokens = batched_probabilities(
        hf_logits(model), tokenizer, prompts, widths, batch_size=args.batch_size, device=device
    )
    elapsed = time.time() - t0
    summary = summarize(probs, labels)
    print(json.dumps(summary.as_dict(), indent=2))
    print(f"{elapsed:.0f}s for {len(prompts)} items, {tokens:,} prompt tokens")

    results = {
        "model": args.model,
        "dataset": meta,
        "chat_template": wrap is not None,
        "n": len(prompts),
        "prompt_tokens": tokens,
        "seconds": round(elapsed, 1),
        "hardware": {
            "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "summary": summary.as_dict(),
        "reliability": [b.__dict__ for b in reliability(probs, labels)],
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    with (out / "predictions.jsonl").open("w") as f:
        for (_qid, req), p, y in zip(requests, probs, labels, strict=True):
            f.write(
                json.dumps({"state": req.state[:200], "p": [round(v, 4) for v in p], "label": y})
                + "\n"
            )
    print(f"wrote {out / 'results.json'} and predictions.jsonl")


if __name__ == "__main__":
    main()
