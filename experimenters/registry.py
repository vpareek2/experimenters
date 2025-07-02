# experimenters/registry.py
"""Global phone-books mapping *string keys* → (class, config_schema).

Updated to use the new typed dataclasses from `experimenters.config.schema`.
"""
from __future__ import annotations

from experimenters.attn import MLA, MHA
from experimenters.ffn import SwiGLU, MoE, GELU
from experimenters.config.schema import MLAConfig, SwiGLUConfig, MoEConfig

# -----------------------------------------------------------------------------
#  Public registries
# -----------------------------------------------------------------------------

ATTN: dict[str, tuple[type, type]] = {
    "MLA": (MLA, MLAConfig),
    "MHA": (MHA, None),
}

FFN: dict[str, tuple[type, type]] = {
    "SwiGLU": (SwiGLU, SwiGLUConfig),
    "MoE": (MoE, MoEConfig),
    "GELU": (GELU, None),
}

__all__ = ["ATTN", "FFN"]
