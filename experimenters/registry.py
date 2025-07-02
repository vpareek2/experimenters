# experimenters/registry.py
"""Global phone-books mapping *string keys* → (class, config_schema).

Updated to use the new typed dataclasses from `experimenters.config.schema`.
"""
from __future__ import annotations

from experimenters.attn import MLA
from experimenters.ffn import SwiGLU, MoE
from experimenters.config.schema import MLAConfig, SwiGLUConfig, MoEConfig

# -----------------------------------------------------------------------------
#  Public registries
# -----------------------------------------------------------------------------

ATTN: dict[str, tuple[type, type]] = {
    "MLA": (MLA, MLAConfig),
}

FFN: dict[str, tuple[type, type]] = {
    "SwiGLU": (SwiGLU, SwiGLUConfig),
    "MoE": (MoE, MoEConfig),
}

__all__ = ["ATTN", "FFN"]
