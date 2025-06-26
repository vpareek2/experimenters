# experimenters/ffn/_parallel.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experimenters.utils.dist import get_world_size, get_rank


class _ColumnParallelLinear(nn.Module):
    """
    Split the output dimension across ranks. No collective needed on forward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        world = get_world_size()
        assert out_features % world == 0, "out_features must divide world size"
        self.out_local = out_features // world
        self.weight = nn.Parameter(torch.empty(self.out_local, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(self.out_local)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,in)  →  (B,T,out_local)
        return F.linear(x, self.weight, self.bias)


class _RowParallelLinear(nn.Module):
    """
    Split the input dimension across ranks. All-reduce on output.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        world = get_world_size()
        assert in_features % world == 0, "in_features must divide world size"
        self.in_local = in_features // world
        self.weight = nn.Parameter(torch.empty(out_features, self.in_local))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        world = get_world_size()
        rank = get_rank()
        xs = x
        if world > 1:
            xs = x[..., rank * self.in_local : (rank + 1) * self.in_local]
        out = F.linear(xs, self.weight)  # (B,T,out)
        if world > 1:
            torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
        if self.bias is not None:
            out += self.bias
        return out
