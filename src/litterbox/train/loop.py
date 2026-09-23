"""Single-device training loop, written by hand rather than delegated.

The loop is short. What it has to get right is mostly *not* the loop:

- **Where the run happens.** CUDA if present, then Apple's GPU, then CPU. Nothing
  below names a device; tensors follow the model and the batch.
- **Precision.** The forward pass runs under ``torch.autocast`` in bf16 while
  parameters, gradients, and optimizer state stay fp32. The model is never
  ``.to(bfloat16)``: that also casts buffers, and is how RoPE's angle tables
  used to get quietly truncated.
- **Which parameters decay.** Matrices do; gains and biases do not. Decaying a
  norm gain pulls it toward zero, which fights the normalisation it exists for.
- **Exact resume.** Batches are a function of ``(seed, step)`` and nothing in the
  model is stochastic, so a checkpoint is the weights, the optimizer state, and
  the step. A run stopped at step 500 and resumed is bit-identical to one that
  never stopped — and there is a test that says so.
- **A validation loss that means something.** Evaluation uses the same fixed
  batches every time, so two numbers from different steps differ because the
  model changed, not because the sample did.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from litterbox.data import PackedDataset, load_data_config, prepare
from litterbox.data.tokenizer import Tokenizer, build_tokenizer

# ------------------------------------------------------------------- config


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptimizerConfig(_Strict):
    type: Literal["adamw"] = "adamw"
    lr: float = Field(gt=0)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0)
    eps: float = 1e-8
    grad_clip: float | None = Field(default=1.0, gt=0)


class ScheduleConfig(_Strict):
    type: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)


class PrecisionConfig(_Strict):
    # float16 is deliberately absent: it needs loss scaling, and bf16 has the
    # range of fp32 so it does not. Every GPU this targets supports bf16.
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    grad_dtype: Literal["float32"] = "float32"


class TrainingConfig(_Strict):
    seed: int = 1337
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    grad_accum_steps: int = Field(default=1, gt=0)
    seq_len: int = Field(gt=0)
    optimizer: OptimizerConfig
    schedule: ScheduleConfig = ScheduleConfig()
    precision: PrecisionConfig = PrecisionConfig()
    device: str = "auto"
    compile: bool = False  # worth it on CUDA; unreliable on Apple's GPU


class DataRef(_Strict):
    config: str  # path to a configs/data/*.yaml


class LoggingConfig(_Strict):
    backend: Literal["jsonl"] = "jsonl"
    out_dir: str = "runs/default"
    log_every: int = Field(default=20, gt=0)
    eval_every: int = Field(default=1000, gt=0)
    eval_batches: int = Field(default=20, gt=0)
    checkpoint_every: int = Field(default=2000, gt=0)


class RunConfig(_Strict):
    training: TrainingConfig
    data: DataRef
    logging: LoggingConfig = LoggingConfig()


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Read and strictly validate a training config.

    ``overrides`` are OmegaConf dotlist entries — ``["training.max_steps=50",
    "logging.out_dir=runs/smoke"]`` — merged over the file before validation.
    """
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return RunConfig.model_validate(OmegaConf.to_container(cfg, resolve=True))


# ------------------------------------------------------------ the small parts


def pick_device(prefer: str = "auto") -> torch.device:
    """CUDA, then Apple's GPU, then CPU — unless told otherwise."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, max_steps: int, peak: float, schedule: ScheduleConfig) -> float:
    """Learning rate for ``step``: linear warmup, then cosine decay to a floor.

    Warmup because Adam's second-moment estimate is garbage for the first few
    hundred steps and a full-size step on a garbage estimate can wreck a run.
    Cosine because it spends longest near the peak and lands gently; the floor
    is nonzero so the last stretch of training still moves.
    """
    if step < schedule.warmup_steps:
        return peak * (step + 1) / schedule.warmup_steps
    if schedule.type == "constant":
        return peak
    span = max(1, max_steps - schedule.warmup_steps)
    progress = min(1.0, (step - schedule.warmup_steps) / span)
    floor = peak * schedule.min_lr_ratio
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into those that decay and those that must not.

    The rule is dimensionality: tensors with two or more axes are weight
    matrices and embedding tables, and decay; one-axis tensors are norm gains
    and biases, and do not. ``parameters()`` already yields a tied
    embedding/LM-head matrix once, so it is decayed once.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _autocast(device: torch.device, precision: PrecisionConfig):
    if precision.dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: PackedDataset,
    *,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    precision: PrecisionConfig | None = None,
    seed: int = 0,
) -> float:
    """Mean loss over a *fixed* set of batches.

    Steps ``0..n_batches-1`` under a fixed seed: every call sees the same
    windows, so a change in this number is a change in the model.
    """
    precision = precision or PrecisionConfig(dtype="float32")
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(n_batches):
        x, y = data.get_batch(batch_size, step=i, seed=seed, device=device)
        with _autocast(device, precision):
            logits = model(x)
        total += F.cross_entropy(logits.float().flatten(0, 1), y.flatten()).item()
    model.train(was_training)
    return total / n_batches


# -------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, model: nn.Module, optimizer, step: int, cfg: RunConfig) -> None:
    """Weights, optimizer state, step, and the config that produced them.

    Written to a temporary name and renamed, so a crash mid-write leaves the
    previous checkpoint intact instead of a truncated file named ``latest``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg.model_dump(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None) -> int:
    """Restore in place and return the step to continue from.

    Loaded onto the CPU first and copied into the live tensors, so a checkpoint
    written on a CUDA machine opens on a laptop and the reverse.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])  # absent on a fresh, step-0 checkpoint
    return int(state["step"])


# ---------------------------------------------------------------- the loop


@dataclass
class TrainResult:
    step: int
    train_loss: float
    val_loss: float | None
    device: str
    history: list[dict[str, Any]] = field(default_factory=list)


def train(
    model: nn.Module,
    cfg: RunConfig,
    *,
    train_data: PackedDataset | None = None,
    val_data: PackedDataset | None = None,
    tokenizer: Tokenizer | None = None,
    resume: bool = True,
    stop_at_step: int | None = None,
    on_eval: Callable[[int, nn.Module], None] | None = None,
    quiet: bool = False,
) -> TrainResult:
    """Train ``model`` according to ``cfg``.

    Args:
        model: any module mapping token ids ``[batch, seq]`` to logits. Built by
            the caller — the loop has no opinion on architecture.
        train_data, val_data: packed datasets. When omitted they are prepared
            from ``cfg.data.config`` (a no-op if the directory is up to date).
        resume: continue from ``<out_dir>/latest.pt`` when it exists.
        stop_at_step: checkpoint and return early at this step, leaving the
            schedule (which is defined over ``max_steps``) untouched. For
            time-limited jobs, and for proving that resuming is exact.
        on_eval: called as ``on_eval(step, model)`` after each evaluation — the
            hook for printing a sample. The model is in eval mode for the call.
    """
    t = cfg.training
    say = (lambda *a, **k: None) if quiet else print
    out_dir = Path(cfg.logging.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    if train_data is None:
        data_cfg = load_data_config(cfg.data.config)
        meta = prepare(data_cfg, progress=not quiet)
        tokenizer = tokenizer or (build_tokenizer(meta.tokenizer) if meta.tokenizer else None)
        train_data = PackedDataset(data_cfg.out_dir, t.seq_len, split="train", tokenizer=tokenizer)
        if val_data is None and "val" in meta.splits:
            val_data = PackedDataset(data_cfg.out_dir, t.seq_len, split="val", tokenizer=tokenizer)

    # An id past the embedding table is an index error on CPU and, on CUDA, an
    # asynchronous device assert that surfaces somewhere unrelated. Check here.
    vocab = getattr(model, "vocab_size", None)
    if vocab is not None and train_data.meta.vocab_size > vocab:
        raise ValueError(
            f"the dataset was packed with a vocabulary of {train_data.meta.vocab_size:,} "
            f"but the model's embedding has {vocab:,} rows"
        )

    limit = getattr(model, "max_seq_len", None)
    if limit is not None and t.seq_len > limit:
        raise ValueError(
            f"training.seq_len={t.seq_len} exceeds the model's max_seq_len={limit}; "
            f"RoPE has no angles for positions past it"
        )

    # ---- device and precision
    device = pick_device(t.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")  # TF32 matmuls: free speed on Ampere+
    torch.manual_seed(t.seed)
    model = model.to(device)  # placement only; precision is autocast's job
    model.train()
    forward = torch.compile(model) if t.compile else model

    optimizer = torch.optim.AdamW(
        param_groups(model, t.optimizer.weight_decay),
        lr=t.optimizer.lr,
        betas=t.optimizer.betas,
        eps=t.optimizer.eps,
    )

    start = 0
    latest = out_dir / "latest.pt"
    if resume and latest.exists():
        start = load_checkpoint(latest, model, optimizer)
        say(f"resumed from {latest} at step {start:,}")

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = t.batch_size * t.grad_accum_steps * t.seq_len
    say(
        f"{n_params / 1e6:.1f}M parameters on {device} | {t.precision.dtype} | "
        f"{tokens_per_step:,} tokens/step | {len(train_data):,} training tokens "
        f"({len(train_data) / tokens_per_step:,.0f} steps/epoch)"
    )

    log_path = out_dir / "log.jsonl"
    history: list[dict[str, Any]] = []
    loss_value, val_loss = float("nan"), None
    window_start, window_tokens = time.perf_counter(), 0

    with log_path.open("a") as log:

        def record(entry: dict[str, Any]) -> None:
            history.append(entry)
            log.write(json.dumps(entry) + "\n")
            log.flush()

        last_step = t.max_steps if stop_at_step is None else min(stop_at_step, t.max_steps)
        for step in range(start, last_step):
            lr = lr_at(step, t.max_steps, t.optimizer.lr, t.schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Gradient accumulation: several micro-batches, one optimizer step.
            # Each micro-batch gets its own data index so none is seen twice,
            # and the loss is divided so the summed gradient is an average.
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for micro in range(t.grad_accum_steps):
                x, y = train_data.get_batch(
                    t.batch_size, step=step * t.grad_accum_steps + micro, seed=t.seed, device=device
                )
                with _autocast(device, t.precision):
                    logits = forward(x)
                # Cross-entropy in fp32: a log-softmax over 50k classes is
                # exactly where bf16's three digits are not enough.
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                (loss / t.grad_accum_steps).backward()
                accumulated += loss.item() / t.grad_accum_steps

            # Always measured, clipped only if asked: max_norm=inf returns the
            # norm and changes nothing.
            max_norm = t.optimizer.grad_clip if t.optimizer.grad_clip is not None else math.inf
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

            loss_value = accumulated
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"loss is {loss_value} at step {step}; stopping")
            # A gradient of exactly zero across a whole model is never a
            # training signal, it is a fault — and it does not show up in the
            # loss, which just stops moving. Seen for real on an oversubscribed
            # Apple GPU, which carried on for twenty steps reporting 0.0.
            if grad_norm == 0.0 or not math.isfinite(grad_norm):
                raise FloatingPointError(
                    f"gradient norm is {grad_norm} at step {step}: the backward pass produced "
                    f"nothing usable. If this is an accelerator under memory pressure, lower "
                    f"training.batch_size and raise training.grad_accum_steps to match."
                )
            optimizer.step()
            window_tokens += tokens_per_step
            done = step + 1

            if done % cfg.logging.log_every == 0 or done == last_step:
                elapsed = time.perf_counter() - window_start
                rate = window_tokens / elapsed if elapsed > 0 else 0.0
                record(
                    {
                        "step": done,
                        "loss": loss_value,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "tokens_per_sec": rate,
                    }
                )
                say(
                    f"step {done:>6,}  loss {loss_value:6.3f}  lr {lr:.2e}  "
                    f"grad {grad_norm:5.2f}  {rate:>9,.0f} tok/s"
                )
                window_start, window_tokens = time.perf_counter(), 0

            if val_data is not None and (done % cfg.logging.eval_every == 0 or done == last_step):
                val_loss = evaluate(
                    model,
                    val_data,
                    batch_size=t.batch_size,
                    n_batches=cfg.logging.eval_batches,
                    device=device,
                    precision=t.precision,
                )
                record({"step": done, "val_loss": val_loss})
                say(f"step {done:>6,}  val  {val_loss:6.3f}")
                if on_eval is not None:
                    model.eval()
                    on_eval(done, model)
                    model.train()
                window_start, window_tokens = time.perf_counter(), 0  # don't bill eval to training

            if done % cfg.logging.checkpoint_every == 0 or done == last_step:
                save_checkpoint(latest, model, optimizer, done, cfg)

    return TrainResult(
        step=max(start, last_step),
        train_loss=loss_value,
        val_loss=val_loss,
        device=str(device),
        history=history,
    )
