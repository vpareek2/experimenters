import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

__all__ = ["MHA"]

class MHA(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True,   # expect (B, T, D)
        )

    def forward(
        self,
        x: Tensor,                 # (B, T, D)
        *_unused,                  # freqs, etc. (ignored)
        cache: Optional[Dict] = None,
    ):  # -> (B, T, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        return out, cache
