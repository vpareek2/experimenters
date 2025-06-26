import torch.nn as nn
from experimenters.embed.token import TokenEmbedding
from experimenters.embed.rope import precompute_rope_freqs
from experimenters.norm import RMSNorm

class TinyConfig:          # simple dataclass stand-in
    dim = 256; n_heads = 4; n_layers = 6; max_seq_len = 256
    qk_rope_dim = 16; latent_dim = 64
    ff_hidden_mult = 4      # hidden_dim = dim * ff_hidden_mult
    ffn_type = "SwiGLU"     # or "ReLUFFN" or "MoE"

def build_transformer(cfg: TinyConfig,
                      *,
                      attn_cls,
                      ffn_cls,
                      norm_cls=RMSNorm,
                      embed_cls=TokenEmbedding):
    model_dim = cfg.dim
    hidden_dim = model_dim * cfg.ff_hidden_mult

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = norm_cls(model_dim)
            self.attn  = attn_cls(model_dim, cfg.n_heads,
                                  latent_dim=cfg.latent_dim,
                                  qk_rope_dim=cfg.qk_rope_dim)
            self.norm2 = norm_cls(model_dim)
            self.ffn   = ffn_cls(model_dim, hidden_dim)

        def forward(self, x, freqs, cache=None):
            a, cache = self.attn(self.norm1(x), freqs, cache)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, cache

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = embed_cls(cfg.vocab_size if hasattr(cfg, "vocab_size") else 256,
                                   model_dim)
            self.freqs = precompute_rope_freqs(cfg.max_seq_len, cfg.qk_rope_dim)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
            self.norm = norm_cls(model_dim)
            self.lm_head = nn.Linear(model_dim, cfg.vocab_size if hasattr(cfg,"vocab_size") else 256, bias=False)
            self.lm_head.weight = self.embed.weight  # weight-tying

        def forward(self, tokens):
            x = self.embed(tokens)
            cache = [None] * len(self.blocks)
            for i, blk in enumerate(self.blocks):
                x, cache[i] = blk(x, self.freqs, cache[i])
            x = self.norm(x)[:, -1]         # logits for last token
            return self.lm_head(x), cache

    return Transformer()
