# experimenters/config/model.py
from dataclasses import dataclass, field
from experimenters.config.attn import MLAConfig
from experimenters.config.ffn  import SwiGLUConfig

@dataclass
class TransformerCfg:
    # global geometry
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 4
    max_seq_len: int = 1024
    vocab_size: int = 129_280

    # which blocks to use
    attn_type: str = "MLA"        # keys for registry
    ffn_type: str  = "SwiGLU"
    moe_layers: int = 0           # last N layers become MoE

    # nested component configs (can be overridden in YAML)
    attn: MLAConfig = field(default_factory=MLAConfig)
    ffn:  SwiGLUConfig = field(default_factory=SwiGLUConfig)
