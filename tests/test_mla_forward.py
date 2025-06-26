# tests/test_mla_forward.py
import torch
from experimenters.attn import MLA

B, T, D = 2, 8, 256
mlp = MLA(dim=D, n_heads=4)        # tiny config
x = torch.randn(B, T, D)
freqs = torch.ones(1024, 8, dtype=torch.complex64)  # dummy RoPE
out, _ = mlp(x, freqs)
assert out.shape == (B, T, D)
