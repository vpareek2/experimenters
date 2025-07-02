# experimenters/registry.py
"""Global phone-books mapping *string keys* → (class, config_schema).

Updated to use the new typed dataclasses from `experimenters.config.schema`.
"""
from __future__ import annotations

from experimenters.attn import MLA
from experimenters.ffn import SwiGLU, MoE
from experimenters.config.schema import MLAConf, SwiGLUConf, MoEConf

# -----------------------------------------------------------------------------
#  Public registries
# -----------------------------------------------------------------------------

ATTN: dict[str, tuple[type, type]] = {
    "MLA": (MLA, MLAConf),
}

FFN: dict[str, tuple[type, type]] = {
    "SwiGLU": (SwiGLU, SwiGLUConf),
    "MoE": (MoE, MoEConf),
}

__all__ = ["ATTN", "FFN"]
