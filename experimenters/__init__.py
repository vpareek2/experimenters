from .config.schema import (
    ModelConfig,
    MLAConfig,
    SwiGLUConfig,
    MoEConfig,
)
from .data.schema import DataConfig
from .model.transformer import build_transformer
from .trainer import Trainer
from .data.loader import load_dataset
from .registry import ATTN, FFN
from . import callbacks as callbacks

__all__ = [
    "ModelConfig", "MLAConfig", "SwiGLUConfig", "MoEConfig", "DataConfig",
    "build_transformer", "Trainer", "load_dataset",
    "ATTN", "FFN",
]

# ---------------------------------------------------------------------
# One-liner convenience: build → Trainer → train
# ---------------------------------------------------------------------
def run(cfg, *, callbacks=None, resume=None, device=None):
    """
    xp.run(cfg)  builds the model, creates a Trainer, and starts training.

    Parameters
    ----------
    cfg        : ModelConfig
    callbacks  : list[Callable] | None   – passed straight to Trainer
    resume     : str | None              – checkpoint dir to load before training
    device     : str | None              – 'cuda', 'cpu', etc.; None = auto-pick

    Returns
    -------
    trainer    : Trainer - so you can inspect or save afterwards
    """
    # automatic registry lookup happens inside build_transformer
    model = build_transformer(cfg)
    trainer = Trainer(model, cfg, device=device, callbacks=callbacks or [])
    if resume:
        trainer.load(resume)
    trainer.train()
    return trainer

__all__ += ["run", "callbacks"]
