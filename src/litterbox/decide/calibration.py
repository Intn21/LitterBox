"""Is the number honest? Calibration measured on groups of answers.

A single answer is right or wrong; "0.7" cannot be right or wrong on its own.
So every measure here works on a pile of answers: group them by what the model
said, compare that to how often the group was right, and report the gap. That
gap is what the threshold in the caller's code depends on, ``act above 0.95``
is only correct if 0.95 means 95%, and nothing in the model checks it.

What is here, and what each one catches:

- **Brier** and **log loss**: per-answer proper scores, the same units the
  training loss uses, so train and eval agree on what "better" means.
- **ECE**, expected calibration error: bin by stated confidence, weight each
  bin's gap by its size, average. Two binnings, because equal-width bins leave
  one bin holding almost everything when a model is mostly confident, and
  equal-mass bins do not.
- **The noise floor**: the ECE a perfectly honest model would show on these
  same predictions, found by drawing labels from the model's own probabilities.
  A bin of eight answers at 0.6 will not land on exactly 4.8 right. A reported
  ECE means nothing without its floor.
- **Coverage at an error budget**, and its area version **AURC**: sort by
  confidence, automate from the top, stop before the error rate crosses the
  budget. This reads the *ordering* of the numbers, which ECE does not, and
  which temperature scaling cannot change.

Inputs are a probability vector per answer (``[n, k]``, or a ragged list of
lists padded here with zeros) and an integer label per answer. Confidence is
the top probability throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor

Scheme = Literal["width", "mass"]


def as_probabilities(p: Sequence[Sequence[float]] | Tensor) -> Tensor:
    """``[n, k]`` float tensor; ragged rows are right-padded with zeros."""
    if isinstance(p, Tensor):
        return p.detach().float().cpu()
    rows = [torch.as_tensor(list(r), dtype=torch.float32) for r in p]
    width = max(len(r) for r in rows)
    out = torch.zeros(len(rows), width)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def as_labels(y: Sequence[int] | Tensor) -> Tensor:
    return torch.as_tensor(y, dtype=torch.long).cpu()


# --- per-answer proper scores -----------------------------------------------------


def brier(p, y) -> float:
    """Mean squared distance between the distribution and the one-hot label; 0 is perfect."""
    p, y = as_probabilities(p), as_labels(y)
    onehot = torch.zeros_like(p)
    onehot[torch.arange(len(y)), y] = 1.0
    return float(((p - onehot) ** 2).sum(-1).mean())


def log_loss(p, y, eps: float = 1e-12) -> float:
    """Mean negative log probability of the label; 0 is perfect."""
    p, y = as_probabilities(p), as_labels(y)
    return float(-p[torch.arange(len(y)), y].clamp(min=eps).log().mean())


def accuracy(p, y) -> float:
    p, y = as_probabilities(p), as_labels(y)
    return float((p.argmax(-1) == y).float().mean())


# --- reliability and ECE ---------------------------------------------------------------


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    confidence: float  # mean stated top probability in the bin
    accuracy: float  # fraction of the bin that was right

    @property
    def gap(self) -> float:
        return abs(self.confidence - self.accuracy)


def reliability(p, y, bins: int = 10, scheme: Scheme = "width") -> list[Bin]:
    """The reliability table: one row per non-empty bin of stated confidence.

    ``"width"`` cuts ``[0, 1]`` into equal intervals; ``"mass"`` cuts the sorted
    confidences into bins of equal count, so no bin can hold everything.
    """
    p, y = as_probabilities(p), as_labels(y)
    conf, pred = p.max(-1)
    hit = (pred == y).float()
    if scheme == "width":
        edges = torch.linspace(0, 1, bins + 1)
        edges[0] = -1.0  # a confidence of exactly 0 belongs to the first bin
    elif scheme == "mass":
        qs = torch.linspace(0, 1, bins + 1)
        edges = torch.quantile(conf, qs)
        edges[0], edges[-1] = -1.0, 1.0
    else:
        raise ValueError(f"scheme must be 'width' or 'mass', got {scheme!r}")
    out = []
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            out.append(Bin(lo, hi, int(sel.sum()), float(conf[sel].mean()), float(hit[sel].mean())))
    return out


def ece(p, y, bins: int = 10, scheme: Scheme = "width") -> float:
    """Expected calibration error: bin gaps weighted by bin size. 0 is perfect."""
    n = len(as_labels(y))
    return sum(b.count * b.gap for b in reliability(p, y, bins, scheme)) / n


def ece_noise_floor(
    p, bins: int = 10, scheme: Scheme = "width", reps: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Mean and standard deviation of the ECE a perfectly calibrated model would show here.

    Labels are drawn from the model's own probabilities ``reps`` times. The
    mean is the floor a real ECE should be compared against; the standard
    deviation says how far above it is still noise.
    """
    p = as_probabilities(p)
    g = torch.Generator().manual_seed(seed)
    values = torch.tensor(
        [ece(p, torch.multinomial(p, 1, generator=g).squeeze(1), bins, scheme) for _ in range(reps)]
    )
    return float(values.mean()), float(values.std())


# --- ordering: coverage and AURC --------------------------------------------------------


def risk_coverage(p, y) -> tuple[Tensor, Tensor]:
    """Error rate among the top-``i`` most confident answers, for every ``i``.

    Returns ``(coverage, risk)``, each ``[n]``: coverage ``i/n`` and the error
    rate among those ``i``. Ties in confidence are broken by original order.
    """
    p, y = as_probabilities(p), as_labels(y)
    conf, pred = p.max(-1)
    order = torch.argsort(conf, descending=True, stable=True)
    wrong = (pred[order] != y[order]).float()
    i = torch.arange(1, len(y) + 1)
    return i / len(y), wrong.cumsum(0) / i


def coverage_at(p, y, budget: float = 0.05) -> float:
    """Largest fraction automatable, from the top of the confidence order, at ≤ ``budget`` error."""
    coverage, risk = risk_coverage(p, y)
    ok = (risk <= budget).nonzero()
    return 0.0 if len(ok) == 0 else float(coverage[ok.max()])


def aurc(p, y) -> float:
    """Area under the risk-coverage curve: mean error rate over every prefix. Lower is better."""
    _, risk = risk_coverage(p, y)
    return float(risk.mean())


# --- a summary, for experiment records --------------------------------------------------


@dataclass(frozen=True)
class Summary:
    n: int
    accuracy: float
    brier: float
    log_loss: float
    ece_width: float
    ece_mass: float
    floor_width: float
    floor_width_std: float
    coverage_at_5: float
    aurc: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(p, y, bins: int = 10, reps: int = 200, seed: int = 0) -> Summary:
    """Every number above in one JSON-ready record."""
    p, y = as_probabilities(p), as_labels(y)
    floor, floor_std = ece_noise_floor(p, bins, "width", reps, seed)
    return Summary(
        n=len(y),
        accuracy=accuracy(p, y),
        brier=brier(p, y),
        log_loss=log_loss(p, y),
        ece_width=ece(p, y, bins, "width"),
        ece_mass=ece(p, y, bins, "mass"),
        floor_width=floor,
        floor_width_std=floor_std,
        coverage_at_5=coverage_at(p, y, 0.05),
        aurc=aurc(p, y),
    )


# --- the picture -----------------------------------------------------------------------


def reliability_diagram(p, y, bins: int = 10, scheme: Scheme = "width", ax=None, title: str = ""):
    """Bars of accuracy per confidence bin against the diagonal, counts above each bar.

    Needs matplotlib (``uv sync --extra eval``); returns the axes.
    """
    import matplotlib.pyplot as plt  # optional dependency, imported here on purpose

    rows = reliability(p, y, bins, scheme)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#6b6a63")
    width = 0.8 / bins
    ax.bar([b.confidence for b in rows], [b.accuracy for b in rows], width=width, alpha=0.85)
    for b in rows:
        ax.text(b.confidence, b.accuracy + 0.02, str(b.count), ha="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("fraction correct")
    ax.set_title(f"{title}  ECE {ece(p, y, bins, scheme):.3f}".strip(), fontsize=9)
    return ax
