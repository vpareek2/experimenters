# experimenters/utils/checkpoint.py
import torch
import json
import os
from typing import Dict, Any


def save_checkpoint(model, optimizer, step: int, path: str, **meta):
    os.makedirs(path, exist_ok=True)  # Ensure the checkpoint directory exists
    torch.save(model.state_dict(), f"{path}/model_state.pt")  # Save model parameters
    if optimizer is not None:
        torch.save(
            optimizer.state_dict(), f"{path}/optim_state.pt"
        )  # Save optimizer state if provided
    meta_out = {"step": step, **meta}  # Combine step and any extra metadata
    with open(f"{path}/meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)  # Save metadata as JSON


def load_checkpoint(model, optimizer, path: str) -> Dict[str, Any]:
    # Load model parameters; strict=False allows for partial loading (e.g., dense↔MoE)
    model.load_state_dict(
        torch.load(f"{path}/model_state.pt", map_location="cpu"), strict=False
    )
    # Load optimizer state if optimizer is provided and checkpoint exists
    if optimizer is not None and os.path.exists(f"{path}/optim_state.pt"):
        optimizer.load_state_dict(
            torch.load(f"{path}/optim_state.pt", map_location="cpu")
        )
    # Load and return metadata
    with open(f"{path}/meta.json") as f:
        return json.load(f)
