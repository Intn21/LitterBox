# Experiments

Every training run that produces a claim gets a folder here. Research repos die
when results live in scattered W&B runs; a dated folder with a config, a
hypothesis, and a `results.json` makes claims reproducible and gives future
contributors worked examples to copy.

## Convention

```
experiments/YYYY-MM-DD-short-slug/
├── config.yaml      # the exact resolved config, not a reference to one
├── notes.md         # hypothesis (before) -> result -> conclusion (after)
└── results.json     # harness output, machine-readable
```

Copy `_template/` to start one.

## Rules

- The **hypothesis is written before the run**. A hypothesis written afterwards
  is a description, and it will always be confirmed.
- The config is **resolved and copied**, not referenced. Configs on `main` change;
  the run that produced these numbers did not.
- **Record the hardware and the versions** — GPU architecture, PyTorch, Triton.
  Version drift in this space produces bugs that look convincingly like research
  findings.
- **Negative results get the same treatment as positive ones.** They are often
  the more useful record.
