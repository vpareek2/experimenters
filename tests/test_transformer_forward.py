import torch
from experimenters.attn import MLA
from experimenters.ffn import SwiGLU
from experimenters.model.transformer import build_transformer, TinyConfig

def test_transformer_forward():
    cfg = TinyConfig()
    model = build_transformer(cfg, attn_cls=MLA, ffn_cls=SwiGLU)
    tok = torch.randint(0, 256, (2, 12))
    logits, cache = model(tok)
    assert logits.shape == (2, 256)
    assert len(cache) == cfg.n_layers
