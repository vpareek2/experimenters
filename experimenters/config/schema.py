from __future__ import annotations
"""Typed configuration schemas for *experimenters*.

Exposed at top‑level import::

    import experimenters as xp
    cfg = xp.ModelConfig(...)

Main classes
------------
* ``ModelConfig`` – global model + training hyper‑params (+ optional ``data``)
* ``MLAConfig``   – attention‑specific knobs (Multi‑Head Latent Attention)
* ``SwiGLUConfig`` & ``MoEConfig`` – feed‑forward variants
* ``DataConfig`` lives in ``experimenters.data.schema`` and can be attached to
  ``ModelConfig.data`` to auto‑load datasets.
"""
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:                # avoid runtime circular import
    from experimenters.data.schema import DataConfig

# ---------------------------------------------------------------------------
#  Attention‑specific config
# ---------------------------------------------------------------------------

@dataclass
class MLAConfig:
    latent_dim: int = 64
    qk_nope_dim: int = 16
    qk_rope_dim: int = 16
    v_head_dim: int = 128
    rope_factor: int = 40
    mscale: float = 1.0

# ---------------------------------------------------------------------------
#  Feed‑forward configs
# ---------------------------------------------------------------------------

@dataclass
class SwiGLUConfig:
    hidden_mult: int = 4

@dataclass
class MoEConfig:
    n_experts: int = 16
    top_k: int = 2
    route_scale: float = 1.0
    n_groups: int = 1
    topk_groups: int = 1
    n_shared_experts: int = 2

# ---------------------------------------------------------------------------
#  Top‑level model / training config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Geometry
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    max_seq_len: int = 1024
    vocab_size: int = 50_257

    # Block choices
    attn_type: Literal["MLA", "MHA"] = "MLA"
    ffn_type:  Literal["SwiGLU", "MoE"] = "SwiGLU"
    moe_layers: int = 0                     # last N layers become MoE

    # Nested per‑block configs
    attn: MLAConfig = field(default_factory=MLAConfig)
    ffn:  SwiGLUConfig | MoEConfig = field(default_factory=SwiGLUConfig)

    # Optional dataset description (used by Trainer)
    data: "DataConfig | None" = None

    # Training hyper‑parameters (used by Trainer)
    lr: float = 3e-4
    batch_size: int = 128
    seq_len: int = 512
    max_steps: int = 1_000
    weight_decay: float = 0.1
    log_interval: int = 50
    grad_clip_norm: float | None = 1.0

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.ffn_type == "MoE":
            assert isinstance(self.ffn, MoEConfig), "ffn must be MoEConfig when ffn_type='MoE'"
