# experimenters/registry.py
from experimenters.attn import MLA, MHA
from experimenters.ffn  import SwiGLU, MoE
from experimenters.config.attn import MLAConfig, MHAConfig
from experimenters.config.ffn  import SwiGLUConfig, MoEConfig

ATTN = {
    "MLA": (MLA, MLAConfig),
}

FFN = {
    "SwiGLU": (SwiGLU, SwiGLUConfig),
    "MoE":    (MoE,    MoEConfig),
}
