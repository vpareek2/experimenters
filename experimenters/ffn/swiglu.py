# experimenters/ffn/swiglu.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from experimenters.ffn._parallel import (
    _ColumnParallelLinear,
    _RowParallelLinear,
)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = _ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.w3 = _ColumnParallelLinear(dim, hidden_dim, bias=False)
        self.w2 = _RowParallelLinear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
