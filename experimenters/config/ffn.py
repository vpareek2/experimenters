# experimenters/config/ffn.py
from dataclasses import dataclass

@dataclass
class SwiGLUConfig:
    hidden_mult: int = 4          # hidden_dim = hidden_mult * model.dim

@dataclass
class MoEConfig:
    n_experts: int = 16
    top_k: int = 2
    route_scale: float = 1.0
