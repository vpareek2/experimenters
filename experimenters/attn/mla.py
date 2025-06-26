# experimenters/attn/mla.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from experimenters.embed.rope import apply_rotary_emb
from experimenters.utils.dist import get_world_size, get_rank


class MLA(nn.Module):
    """
    Multi-Head Latent Attention with automatic tensor-parallel support.

    Parameters
    ----------
    dim            : model hidden size
    n_heads        : global number of heads (must divide world_size)
    latent_dim     : r  (compressed K/V dimension)
    qk_nope_dim    : per-head dims that skip RoPE
    qk_rope_dim    : per-head dims that get RoPE
    v_head_dim     : per-head value dimension
    original_ctx   : original context window size (for scaling)
    rope_theta     : RoPE theta (unused here, for compatibility)
    rope_factor    : context extension factor (for scaling)
    mscale         : additional scaling multiplier
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        latent_dim: int = 128,
        qk_nope_dim: int = 16,
        qk_rope_dim: int = 16,
        v_head_dim: int = 128,
        original_ctx: int = 4096,
        rope_theta: float = 10000.0,
        rope_factor: float = 40,
        mscale: float = 1.0,
    ):
        super().__init__()

        # --- distributed context ------------------------------------------------
        self.world_size = get_world_size()
        self.rank = get_rank()

        assert (
            n_heads % self.world_size == 0
        ), f"n_heads ({n_heads}) must divide world_size ({self.world_size})"

        self.n_heads = n_heads
        self.n_local_heads = n_heads // self.world_size
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.qk_head_dim = qk_nope_dim + qk_rope_dim
        self.v_head_dim = v_head_dim

        # --- projections --------------------------------------------------------
        self.q_proj = nn.Linear(dim, self.n_local_heads * self.qk_head_dim, bias=False)
        self.u_proj = nn.Linear(dim, latent_dim, bias=False)
        self.kv_up = nn.Linear(
            latent_dim,
            self.n_local_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False,
        )
        # row-sharded output: input is local heads, output is full model dim
        self.o_proj = nn.Linear(self.n_local_heads * self.v_head_dim, dim, bias=False)

        self.scale = self.qk_head_dim**-0.5
        # YaRN extra scaling when context > original window
        if rope_factor > 1 and self.qk_rope_dim:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale *= mscale * mscale

    # ---------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # (B, T, dim)
        freqs_cis: torch.Tensor,  # (max_T, qk_rope_dim//2) complex
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        B, T, _ = x.shape

        # 1) project Q and latent
        q = self.q_proj(x)  # (B, T, H_local*qk_head)
        latent = self.u_proj(x)  # (B, T, r)

        # 2) concat latent cache
        if cache is not None:
            latent = torch.cat([cache["latent"], latent], dim=1)
        cache_next = {"latent": latent}

        # 3) up-project K,V for local heads only
        kv = self.kv_up(latent)  # (B, T_tot, H_local*(qk_head+v_head))
        kv = kv.view(B, -1, self.n_local_heads, self.qk_head_dim + self.v_head_dim)
        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
        k, v = k.contiguous(), v.contiguous()

        # 4) reshape Q to (B, H_local, T, qk_head)
        def _split_q(t):
            B_, S, _ = t.shape
            return t.view(B_, S, self.n_local_heads, self.qk_head_dim).transpose(1, 2)

        q = _split_q(q)
        # k, v already (B, T_tot, H_local, qk_head/v_head), need (B, H_local, T_tot, ...)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 5) NOPE / RoPE split
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        k_nope, k_rope = torch.split(k, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # 6) apply RoPE to *_rope slices
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        k_rope = apply_rotary_emb(k_rope, freqs_cis)

        # 7) attention
        scores = (
            q_nope @ k_nope.transpose(-2, -1) + q_rope @ k_rope.transpose(-2, -1)
        ) * self.scale
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        out = probs @ v  # (B, H_local, T, v_head_dim)

        # 8) merge local heads and output proj
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, self.n_local_heads * self.v_head_dim)
        )
        out = self.o_proj(out)  # (B, T, dim)

        # 9) all-reduce to combine row-sharded outputs
        if self.world_size > 1:
            torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)

        return out, cache_next
