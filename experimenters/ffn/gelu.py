import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = dim * hidden_mult
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
