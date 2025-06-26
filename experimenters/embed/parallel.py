# experimenters/embed/parallel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from experimenters.utils.dist import get_rank, get_world_size


class ParallelEmbedding(nn.Module):
    """
    Row-sharded embedding: each rank owns a contiguous slice of the
    vocabulary. Forward reduces the partial outputs with all-reduce.

    Layout
    --------------------------------------------------
    global vocab  = [0 .. V-1]
    local slice   = [rank*V/p  ..  (rank+1)*V/p  - 1]
    --------------------------------------------------
    """

    def __init__(self, vocab_size: int, dim: int, init_std: float = 0.02):
        super().__init__()
        world_size = get_world_size()
        assert (
            vocab_size % world_size == 0
        ), f"vocab_size ({vocab_size}) must be divisible by world_size ({world_size})"

        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        self.rank = get_rank()

        self.part_size = vocab_size // world_size
        self.vocab_start = self.rank * self.part_size
        self.vocab_end = self.vocab_start + self.part_size

        # Local weights
        self.weight = nn.Parameter(torch.empty(self.part_size, dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Algorithm:
            1. Shift token IDs so they index local rows
            2. Embed; rows that belong to other ranks become zeros
            3. All-reduce to sum partial results (equivalent to all-gather)
        """
        if self.world_size == 1:
            return F.embedding(token_ids, self.weight)

        # Mask tokens not in this shard
        mask = (token_ids < self.vocab_start) | (token_ids >= self.vocab_end)
        local_ids = token_ids - self.vocab_start
        local_ids = local_ids.masked_fill(mask, 0)

        # Embed
        embeddings = F.embedding(
            local_ids, self.weight
        )  # (B, T, dim); wrong rows are garbage
        embeddings.masked_fill_(mask.unsqueeze(-1), 0.0)  # Zero them

        # All-reduce
        dist.all_reduce(
            embeddings, op=dist.ReduceOp.SUM
        )  # Sum == gather because shards are disjoint

        return embeddings
