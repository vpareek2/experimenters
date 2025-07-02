import torch
from experimenters.attn import MLA
from experimenters.ffn import SwiGLU
from experimenters.model.transformer import build_transformer
from experimenters.config import ModelConf


def test_transformer_forward():
    cfg = ModelConf(max_seq_len=16, vocab_size=256)  # small for unit test speed
    model = build_transformer(cfg, attn_cls=MLA, ffn_cls=SwiGLU)
    tok = torch.randint(0, cfg.vocab_size, (2, 12))
    logits, cache = model(tok)
    assert logits.shape == (2, cfg.vocab_size)
    assert len(cache) == cfg.n_layers
