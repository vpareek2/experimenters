# experimenters/__init__.py
from importlib import import_module as _imp

# -- configs ----------------------------
from .config.schema import (
    ModelConfig,
    MLAConfig,
    SwiGLUConfig,
    MoEConfig,
    DataConfig,
)

# -- builder & trainer ------------------
from .model.transformer import build_transformer
from .trainer import Trainer

# -- data helpers -----------------------
from .data.loader import load_dataset    # once we add it

# -- registries -------------------------
from .registry import ATTN, FFN

__all__ = [
    # configs
    "ModelConfig", "MLAConfig", "SwiGLUConfig", "MoEConfig", "DataConfig",
    # workflow
    "build_transformer", "Trainer", "load_dataset",
    # advanced
    "ATTN", "FFN",
]
