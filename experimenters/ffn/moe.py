# experimenters/ffn/moe.py
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.distributed as dist

from experimenters.ffn.swiglu import SwiGLU
from experimenters.utils.dist import get_world_size, get_rank


class _Expert(nn.Module):
    """Single routed expert = SwiGLU FFN."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class _Gate(nn.Module):
    """
    DeepSeek-style bias-balancing top-k gate with optional group pruning.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        *,
        n_groups: int = 1,
        topk_groups: int = 1,
        score_func: str = "softmax",  # "softmax" | "sigmoid"
        route_scale: float = 1.0,
    ):
        super().__init__()
        assert score_func in ("softmax", "sigmoid")
        self.top_k = top_k
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.score_func = score_func
        self.route_scale = route_scale

        self.linear = nn.Linear(dim, n_experts, bias=False)
        # Bias row only used by DeepSeek for very-large dims; kept generic.
        self.bias = nn.Parameter(torch.zeros(n_experts))

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x – (N, dim) flattened batch·seq tokens

        Returns:
          weights – (N, top_k)   mixing coefficients (scaled, same dtype as x)
          indices – (N, top_k)   chosen expert indices  (int64)
        """
        logits = self.linear(x)  # (N, E)

        # 1) base activation on pre-bias logits
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:  # sigmoid
            scores = logits.sigmoid()

        # 2) add bias (after activation) if present
        scores = scores + self.bias

        # 3) optional group pruning
        if self.n_groups > 1:
            N, E = scores.shape
            scores_g = scores.view(N, self.n_groups, -1)  # (N, G, E/G)
            if self.score_func == "softmax":
                group_metric = scores_g.amax(dim=-1)  # (N,G)
            else:
                group_metric = scores_g.topk(2, dim=-1).values.sum(-1)
            top_groups = group_metric.topk(self.topk_groups, dim=-1).indices
            mask = torch.ones_like(group_metric, dtype=torch.bool)
            mask.scatter_(1, top_groups, False)
            scores_g = scores_g.masked_fill(mask.unsqueeze(-1), float("-inf"))
            scores = scores_g.flatten(1)  # (N,E)

        # 4) token-level top-k
        topv, topi = scores.topk(self.top_k, dim=-1)  # (N,k)
        weights = logits.gather(1, topi)  # logits for chosen experts
        if self.score_func == "sigmoid":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights.type_as(x), topi


class MoE(nn.Module):
    """
    DeepSeek-V3 MoE layer: routed experts + always-on shared experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        n_experts: int,
        top_k: int = 2,
        n_groups: int | None = 1,
        topk_groups: int | None = 1,
        route_scale: float = 1.0,
        n_shared_experts: int = 2,
        **_,
    ):
        super().__init__()
        world = get_world_size()
        rank = get_rank()
        assert n_experts % world == 0, "n_experts must divide world size"
        self.dim = dim
        self.n_local = n_experts // world
        self.offset = rank * self.n_local

        # ---------------- gate ----------------
        self.gate = _Gate(
            dim,
            n_experts,
            top_k,
            n_groups=n_groups or 1,
            topk_groups=topk_groups or 1,
            score_func="softmax",
            route_scale=route_scale,
        )

        # ---------------- experts -------------
        self.experts = nn.ModuleList(
            [_Expert(dim, hidden_dim) for _ in range(self.n_local)]
        )
        self.shared_experts = nn.ModuleList(
            [_Expert(dim, hidden_dim) for _ in range(n_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, D)       — works on CPU or GPU
        """
        B, T, D = x.shape
        flat = x.view(-1, D)  # (N, D) with N = B*T

        weights, indices = self.gate(flat)  # (N, k)

        Ntokens = flat.size(0)
        expected = Ntokens * self.gate.top_k / (self.n_local * get_world_size())
        capacity = math.ceil(expected * 1.25)

        out = torch.zeros_like(flat)

        offset = self.offset  # global id of local expert 0

        # loop over *local* experts only
        for local_id, expert in enumerate(self.experts):
            gid = offset + local_id  # global expert id
            mask_tok = (indices == gid).any(dim=-1)  # (N,)
            if not mask_tok.any():
                continue

            token_idx = mask_tok.nonzero(as_tuple=False).squeeze(1)
            cols = (indices[token_idx] == gid).int().argmax(dim=-1)
            w_eid = weights[token_idx, cols]  # (M,)

            if w_eid.size(0) > capacity:
                keep = w_eid.topk(capacity).indices
                token_idx = token_idx[keep]
                w_eid = w_eid[keep]

            out[token_idx] += expert(flat[token_idx]) * w_eid[:, None]

        # gather routed outputs from all ranks
        if get_world_size() > 1 and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM)

        # shared experts (replicated on every rank)
        if self.shared_experts:
            shared = sum(e(flat) for e in self.shared_experts) / len(
                self.shared_experts
            )
            out += shared

        return out.view(B, T, D)
