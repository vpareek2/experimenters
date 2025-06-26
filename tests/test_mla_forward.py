import torch
from experimenters.attn import MLA

def test_mla_forward_shape():
    B, T, D = 2, 8, 256
    mla = MLA(dim=D, n_heads=4)
    x = torch.randn(B, T, D)
    freqs = torch.ones(1024, 8, dtype=torch.complex64)  # dummy RoPE
    out, _ = mla(x, freqs)
    assert out.shape == (B, T, D)
