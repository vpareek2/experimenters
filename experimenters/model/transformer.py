import torch.nn as nn
from experimenters.embed.token import TokenEmbedding
from experimenters.embed.rope import precompute_rope_freqs
from experimenters.norm import RMSNorm
from experimenters.registry import ATTN, FFN
from experimenters.config.model import TransformerCfg

def build_transformer(cfg: TransformerCfg):
    attn_cls, _attn_cfg_type = ATTN[cfg.attn_type]
    ffn_cls,  _ffn_cfg_type  = FFN[cfg.ffn_type]

    # layer-by-layer selection for FFN (e.g., MoE in last N layers)
    def choose_ffn(layer_idx):
        if hasattr(cfg, "moe_layers") and cfg.moe_layers > 0:
            if layer_idx >= cfg.n_layers - cfg.moe_layers:
                return FFN["MoE"][0], cfg.ffn  # MoE plus its cfg
        return ffn_cls, cfg.ffn

    model_dim = cfg.dim
    hidden_dim = model_dim * cfg.ffn.hidden_mult

    class Block(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.norm1 = RMSNorm(model_dim)
            self.attn = attn_cls(
                model_dim,
                cfg.n_heads,
                **cfg.attn.__dict__
            )
            self.norm2 = RMSNorm(model_dim)
            sel_cls, sel_cfg = choose_ffn(layer_idx)
            self.ffn = sel_cls(model_dim, hidden_dim, **sel_cfg.__dict__)

        def forward(self, x, freqs, cache=None):
            a, cache = self.attn(self.norm1(x), freqs, cache)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, cache

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = cfg.vocab_size if hasattr(cfg, "vocab_size") else 256
            self.embed = TokenEmbedding(vocab_size, model_dim)
            self.freqs = precompute_rope_freqs(cfg.max_seq_len, cfg.qk_rope_dim)
            self.blocks = nn.ModuleList([Block(i) for i in range(cfg.n_layers)])
            self.norm = RMSNorm(model_dim)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight  # weight-tying

        def forward(self, tokens):
            x = self.embed(tokens)
            cache = [None] * len(self.blocks)
            for i, blk in enumerate(self.blocks):
                x, cache[i] = blk(x, self.freqs, cache[i])
            x = self.norm(x)[:, -1]         # logits for last token
            return self.lm_head(x), cache

    return Transformer()
