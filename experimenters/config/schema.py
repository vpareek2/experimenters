from __future__ import annotations

"""Typed configuration schemas for *experimenters*.

These dataclasses replace the old YAML→dacite plumbing.  Each block has its
own schema so users get IDE autocompletion and mypy checking.

Usage
-----
>>> from experimenters.config.schema import ModelConf, MLAConf, MoEConf
>>> cfg = ModelConf(dim=512, attn=MLAConf(latent_dim=128), ffn=MoEConf(n_experts=32))

The resulting `cfg` object can be passed straight to `build_transformer(cfg)`.
"""

from dataclasses import dataclass, field
from typing import Literal

# -----------------------------------------------------------------------------
#  Attention-specific knobs
# -----------------------------------------------------------------------------

@dataclass
class MLAConf:
    """Configuration for Multi-Head Latent Attention (MLA)."""

    latent_dim: int = 64                # r – compressed K/V size
    qk_nope_dim: int = 16              # dims that skip RoPE
    qk_rope_dim: int = 16              # dims that get RoPE
    v_head_dim: int = 128
    rope_factor: int = 40              # YaRN scaling factor
    mscale: float = 1.0               # additional multiplier

    def __post_init__(self):
        assert self.qk_nope_dim + self.qk_rope_dim > 0, "qk dims must be > 0"


# -----------------------------------------------------------------------------
#  Feed-forward Family
# -----------------------------------------------------------------------------

@dataclass
class SwiGLUConf:
    hidden_mult: int = 4               # FFN hidden size = hidden_mult × model.dim


@dataclass
class MoEConf:
    n_experts: int = 16
    top_k: int = 2
    route_scale: float = 1.0
    n_groups: int = 1
    topk_groups: int = 1
    n_shared_experts: int = 2

    def __post_init__(self):
        assert self.n_experts >= self.top_k, "top_k cannot exceed n_experts"


# -----------------------------------------------------------------------------
#  Model-level geometry & block selection
# -----------------------------------------------------------------------------

@dataclass
class ModelConf:
    """Top-level transformer configuration."""

    # Geometry
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    max_seq_len: int = 1024
    vocab_size: int = 50_257

    # Which block types to use
    attn_type: Literal["MLA"] = "MLA"
    ffn_type: Literal["SwiGLU", "MoE"] = "SwiGLU"
    moe_layers: int = 0                # last N layers become MoE

    # Nested configs – default factories mean they are fully-populated by default
    attn: MLAConf = field(default_factory=MLAConf)
    ffn:  SwiGLUConf | MoEConf = field(default_factory=SwiGLUConf)

    # Trainer/runtime hints (optional at build time)
    lr: float = 3e-4
    batch_size: int = 128
    seq_len: int = 512
    max_steps: int = 1_000
    weight_decay: float = 0.1

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        if self.ffn_type == "MoE":
            assert isinstance(self.ffn, MoEConf), "ffn field must be MoEConf when ffn_type='MoE'"
        else:
            assert isinstance(self.ffn, SwiGLUConf), "ffn field must be SwiGLUConf when ffn_type='SwiGLU'"
