from __future__ import annotations
"""Built‑in training callbacks.

Callbacks are lightweight callables that receive ``(trainer, loss)`` at the
end of every batch.  They can log, checkpoint, or add auxiliary behaviour.

Example usage
-------------
    import experimenters as xp

    cfg = xp.ModelConfig(...)
    xp.run(cfg, callbacks=[xp.callbacks.PrintLoss(every=10),
                           xp.callbacks.CheckpointSaver(every=100)])
"""
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
#  Base protocol – anything __call__(trainer, loss) qualifies
# ---------------------------------------------------------------------------
class Callback(Protocol):
    def __call__(self, trainer, loss): ...

# ---------------------------------------------------------------------------
#  1. Console logger
# ---------------------------------------------------------------------------
class PrintLoss:
    """Print loss to stdout every *N* steps."""

    def __init__(self, every: int = 10):
        self.every = every

    def __call__(self, trainer, loss):
        if trainer.step % self.every == 0:
            tps = trainer.meter.rate / 1e3
            print(f"step {trainer.step:06d}  loss {loss.item():.4f}  {tps:.1f} Ktok/s")

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
# public surface
# ---------------------------------------------------------------------------
__all__ = ["Callback", "PrintLoss", "CheckpointSaver", "WandbLogger"]
