import torch
from experimenters.norm import RMSNorm

def _check(norm_cls):
    norm = norm_cls(dim=32)
    x = torch.randn(4, 10, 32)
    out = norm(x)
    assert out.shape == x.shape

def test_rmsnorm():  _check(RMSNorm)
