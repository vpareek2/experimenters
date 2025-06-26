# experimenters/embed/token.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """
    Single-GPU learnable embedding matrix.
    """

    def __init__(self, vocab_size: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (B, T) int64
        returns   : (B, T, dim)
        """
        return F.embedding(token_ids, self.weight)
