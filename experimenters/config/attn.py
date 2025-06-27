# experimenters/config/attn.py
from dataclasses import dataclass

# --- MLA-specific knobs --------------------------------------------------
@dataclass
class MLAConfig:
    latent_dim: int = 64          # r
    qk_rope_dim: int = 16
    qk_nope_dim: int = 16
    v_head_dim: int = 128
    rope_factor: int = 40         # YaRN long-context boost
    mscale: float = 1.0
