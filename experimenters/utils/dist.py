# experimenters/utils/dist.py
import torch.distributed as dist

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
