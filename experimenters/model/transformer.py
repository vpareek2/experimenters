from __future__ import annotations
"""Transformer builder wired for the new *typed config* system.

`build_transformer(cfg, attn_cls, ffn_cls)` returns a ready-to-train
`nn.Module` configured via `experimenters.config.schema.ModelConf`.
"""

import torch.nn as nn

from experimenters.embed.token import TokenEmbedding
from experimenters.embed.rope import precompute_rope_freqs
from experimenters.norm import RMSNorm
from experimenters.registry import ATTN, FFN
from experimenters.config.schema import ModelConfig, MoEConfig, SwiGLUConfig

# -----------------------------------------------------------------------------
#  Helper: hidden dim calculation
# -----------------------------------------------------------------------------

def _hidden_dim(model_dim: int, ffn_cfg: SwiGLUConfig | MoEConfig):
    if isinstance(ffn_cfg, SwiGLUConfig):
        return model_dim * ffn_cfg.hidden_mult
    # fallback multiplier for MoE when nothing else specified
    return model_dim * 4

# -----------------------------------------------------------------------------
#  Public factory
# -----------------------------------------------------------------------------

def build_transformer(cfg: ModelConfig, *, attn_cls=None, ffn_cls=None):
    """Create a `nn.Module` according to *cfg* using the supplied or looked-up block classes."""

    # --------------------------------------------------
    #  NEW: automatic lookup if the caller passes None
    # --------------------------------------------------
    if attn_cls is None or ffn_cls is None:
        attn_cls, _ = ATTN[cfg.attn_type]
        ffn_cls,  _ = FFN [cfg.ffn_type]

    model_dim = cfg.dim
    hid_dim = _hidden_dim(model_dim, cfg.ffn)

    # ------------- per‑layer FFN selector -----------------
    def choose_ffn(layer_idx: int):
        if cfg.ffn_type == "MoE" or (
            cfg.moe_layers and layer_idx >= cfg.n_layers - cfg.moe_layers
        ):
            return FFN["MoE"][0], cfg.ffn  # class, cfg
        return ffn_cls, cfg.ffn

    # =====================================================
    #  Block definition
    # =====================================================
    class Block(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.norm1 = RMSNorm(model_dim)
            # ---------------- attention -----------------
            attn_kwargs = cfg.attn.__dict__ if cfg.attn_type == "MLA" else {}
            self.attn = attn_cls(
                model_dim,
                cfg.n_heads,
                **attn_kwargs,
            )
            self.norm2 = RMSNorm(model_dim)

            sel_cls, sel_cfg = choose_ffn(layer_idx)
            if isinstance(sel_cfg, SwiGLUConfig):
                # SwiGLU signature: (dim, hidden_dim)
                self.ffn = sel_cls(model_dim, hid_dim)
            else:
                # MoE signature: (dim, hidden_dim, **extra)
                self.ffn = sel_cls(model_dim, hid_dim, **sel_cfg.__dict__)

        def forward(self, x, freqs, cache=None):
            a, cache = self.attn(self.norm1(x), freqs, cache)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, cache

    # =====================================================
    #  Outer Transformer module
    # =====================================================
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = TokenEmbedding(cfg.vocab_size, model_dim)
            self.freqs = precompute_rope_freqs(cfg.max_seq_len, cfg.attn.qk_rope_dim)
            self.blocks = nn.ModuleList([Block(i) for i in range(cfg.n_layers)])
            self.norm = RMSNorm(model_dim)
            self.lm_head = nn.Linear(model_dim, cfg.vocab_size, bias=False)
            # weight tying
            self.lm_head.weight = self.embed.weight

        def forward(self, tokens):
            x = self.embed(tokens)
            cache = [None] * len(self.blocks)
            for i, blk in enumerate(self.blocks):
                x, cache[i] = blk(x, self.freqs, cache[i])
            x = self.norm(x)[:, -1]
            return self.lm_head(x), cache

    return Transformer()
