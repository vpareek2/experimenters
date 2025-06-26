import torch
from experimenters.ffn import SwiGLU

def _check(module_cls):
    ffn = module_cls(dim=64, hidden_dim=128)
    x = torch.randn(4, 10, 64)
    out = ffn(x)
    assert out.shape == x.shape

def test_swiglu_ffn():
    _check(SwiGLU)
