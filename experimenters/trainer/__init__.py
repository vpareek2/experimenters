# experimenters/trainer/__init__.py
"""Minimal Trainer (single GPU, AMP, checkpoints).

Now accepts an optional **callbacks** list so higher‑level helpers like
``xp.run`` can pass logging callbacks, but we keep behaviour otherwise
unchanged.
"""
from __future__ import annotations

import time
from typing import Callable, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from experimenters.utils.checkpoint import save_checkpoint
from experimenters.utils.metrics import ThroughputMeter
from experimenters.data.loader import load_dataset

if TYPE_CHECKING:
    from experimenters.config.schema import ModelConfig


class Trainer:
    """Single‑GPU training loop with AMP and basic logging."""

    def __init__(
        self,
        model: nn.Module,
        cfg: "ModelConfig",
        *,
        device: str | None = None,
        callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.cfg   = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.callbacks = callbacks or []

        # -----------------------------------------------------------------
        # Dataset selection: cfg.data preferred, else legacy cfg.dataset_cls
        # -----------------------------------------------------------------
        if getattr(cfg, "data", None):
            ds: IterableDataset = load_dataset(cfg.data, cfg)  # type: ignore[arg-type]
        elif getattr(cfg, "dataset_cls", None):
            ds = cfg.dataset_cls(cfg)
        else:
            raise ValueError("Trainer requires cfg.data or cfg.dataset_cls")

        num_workers = getattr(cfg, "num_workers", 0)
        self.loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            pin_memory=torch.cuda.is_available(),  # safe on CPU/MPS
            num_workers=num_workers,
        )

        # -----------------------------------------------------------------
        # Optimiser & loss
        # -----------------------------------------------------------------
        self.optim = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.meter = ThroughputMeter()
        self.step  = 0

    # ---------------------------------------------------------------------
    def train(self):
        start = time.time()
        tokens_target = (
            self.cfg.max_steps * self.cfg.batch_size * self.cfg.seq_len
            if self.cfg.max_steps else float("inf")
        )

        for batch in self.loader:
            batch = batch.to(self.device, non_blocking=True)
            logits, _ = self.model(batch)
            loss = self.loss_fn(logits, batch[:, -1])

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(self.cfg, "grad_clip_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.grad_clip_norm
                )
            self.optim.step()

            self.meter.update(batch.numel())
            self.step += 1

            for cb in self.callbacks:
                cb(self, loss)

            if self.step >= self.cfg.max_steps or self.meter.counter >= tokens_target:
                break

        elapsed = time.time() - start
        print(f"finished {self.step} steps in {elapsed/60:.1f} min")

    # ---------------------------------------------------------------------
    def save(self, path: str):
        save_checkpoint(self.model, self.optim, self.step, path)
