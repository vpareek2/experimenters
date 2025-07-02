
from pathlib import Path
from typing import Protocol

import torch
# ---------------------------------------------------------------------------
#  Base protocol – anything __call__(trainer, loss) qualifies
# ---------------------------------------------------------------------------
class Callback(Protocol):
    def __call__(self, trainer, loss): ...

# ---------------------------------------------------------------------------
#  1. Console logger
# ---------------------------------------------------------------------------
class PrintLoss:
    def __init__(self, every: int = 10, show_speed: bool = True):
        self.every, self.show_speed = every, show_speed

    def __call__(self, trainer, loss):
        if trainer.step % self.every:
            return

        # base items
        msg = [
            f"step: {trainer.step:06d}",
            f"loss: {loss.item():.4f}",
        ]

        if self.show_speed:
            tps = trainer.meter.rate / 1e3
            msg.append(f"{tps:4.1f}K tok/s")

        # append any extra metrics other callbacks stored
        for k, v in trainer.metrics.items():
            msg.append(f"{k} {v:.4f}")

        print("  ".join(msg))
        trainer.metrics.clear()          # reset for next batch



# ---------------------------------------------------------------------------
# 2. Periodic checkpoint saver
# ---------------------------------------------------------------------------
class CheckpointSaver:
    """Save ``model_state.pt`` & ``optim_state.pt`` every *every* steps."""

    def __init__(self, every: int = 1000, path: str | Path = "ckpt"):
        self.every = every
        self.path  = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __call__(self, trainer, loss):
        if trainer.step % self.every == 0 and trainer.step > 0:
            subdir = self.path / f"step-{trainer.step:06d}"
            trainer.save(str(subdir))

# ---------------------------------------------------------------------------
#  3. Placeholder for future wandb/CSV loggers
# ---------------------------------------------------------------------------
class WandbLogger:
    """No‑op placeholder. Real implementation would import wandb lazily."""

    def __init__(self, project: str = "experimenters-demo", **kw):
        self.enabled = False
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project, **kw)
            self.enabled = True
        except ImportError:
            print("wandb not installed; WandbLogger disabled.")

    def __call__(self, trainer, loss):
        if not self.enabled:
            return
        self.wandb.log({"loss": loss.item(), "step": trainer.step}, step=trainer.step)

# ---------------------------------------------------------------------------
# 5. LRWarmupCosine – linear warm-up then cosine decay
# ---------------------------------------------------------------------------
import math
class LRWarmupCosine:
    def __init__(self, warmup: int = 200, total: int | None = None):
        self.warmup = warmup
        self.total  = total          # default to cfg.max_steps at first call
    def __call__(self, trainer, _loss):
        if self.total is None:
            self.total = trainer.cfg.max_steps
        step, base_lr = trainer.step, trainer.cfg.lr
        if step <= self.warmup:
            lr = base_lr * step / max(1, self.warmup)
        else:
            pct = (step - self.warmup) / max(1, self.total - self.warmup)
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * pct))
        for g in trainer.optim.param_groups:
            g["lr"] = lr

# ---------------------------------------------------------------------------
# 6. EvalLoop – run a validation loader every N steps
# ---------------------------------------------------------------------------
import torch

class EvalLoop:
    """Run a validation pass every *every* steps, store avg loss."""

    def __init__(self, val_loader, every: int = 500, max_batches: int = 20):
        self.val_loader, self.every, self.max_batches = val_loader, every, max_batches

    @torch.no_grad()
    def __call__(self, trainer, _loss):
        if trainer.step % self.every:
            return

        model, loss_fn = trainer.model, trainer.loss_fn
        model.eval()
        tot, n = 0.0, 0

        for batch in self.val_loader:
            batch = batch.to(trainer.device, non_blocking=True)
            logits, _ = model(batch)
            tot += loss_fn(logits, batch[:, -1]).item()
            n += 1
            if n >= self.max_batches:
                break

        val_loss = tot / n
        trainer.metrics["val_loss"] = val_loss     # Picked up by PrintLoss
        trainer.last_val_loss = val_loss           # EarlyStop can read it
        model.train()


class GradNormLogger:
    def __init__(self, every: int = 10):
        self.every = every

    def __call__(self, trainer, _loss):
        if trainer.step % self.every:
            return
        total = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        trainer.metrics["grad_norm"] = total ** 0.5



class EarlyStop:
    def __init__(self, patience:int = 5):
        self.patience, self.best = patience, float("inf")
        self.bad = 0
    def __call__(self, tr, _loss):
        cur = getattr(tr, "last_val_loss", None)
        if cur is None: return          # need EvalLoop to set this attr
        if cur < self.best:
            self.best, self.bad = cur, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"Early-stop at step {tr.step}")
                raise SystemExit

class TBLogger:
    def __init__(self, logdir="runs/exp", flush=100):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(logdir)
            self.enabled = True
            self.flush = flush
        except ImportError:
            print("tensorboard not found; TBLogger disabled.")
            self.enabled = False
    def __call__(self, tr, loss):
        if not self.enabled: return
        self.tb.add_scalar("train/loss",  loss.item(),  tr.step)
        self.tb.add_scalar("train/lr",    tr.optim.param_groups[0]['lr'], tr.step)
        self.tb.add_scalar("train/tok_s", tr.meter.rate, tr.step)
        if tr.step % self.flush == 0:
            self.tb.flush()


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------
__all__ = ["Callback", "PrintLoss", "CheckpointSaver", "WandbLogger", "LRWarmupCosine", "EvalLoop", "GradNormLogger", "EarlyStop", "TBLogger"]
