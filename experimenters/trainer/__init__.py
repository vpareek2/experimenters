# experimenters/trainer/__init__.py
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experimenters.utils.checkpoint import save_checkpoint
from experimenters.utils.metrics import ThroughputMeter
from experimenters.config import ModelConfig

class Trainer:
    """
    Minimal, clean training loop.

    Required in cfg:
        lr, batch_size, seq_len, max_steps OR max_minutes
        dataset_cls (callable returning an IterableDataset)
    """

    def __init__(self,
                 model: nn.Module,
                 cfg: ModelConfig,
                 *,
                 device: str | None = None):
        self.model = model
        self.cfg   = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # --- dataset --------------------------------------------------------
        ds = cfg.dataset_cls(cfg)               # user supplies dataset factory
        self.loader = DataLoader(ds,
                                 batch_size=cfg.batch_size,
                                 pin_memory=True,
                                 num_workers=2)

        # --- optimizer & loss ----------------------------------------------
        self.optim = torch.optim.AdamW(model.parameters(),
                                       lr=cfg.lr,
                                       betas=(0.9, 0.95),
                                       weight_decay=cfg.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        # --- misc -----------------------------------------------------------
        self.meter = ThroughputMeter()
        self.step  = 0

    # ---------------------------------------------------------------------
    def train(self):
        start = time.time()
        tokens_target = self.cfg.max_steps * self.cfg.batch_size * self.cfg.seq_len \
                        if self.cfg.max_steps else float("inf")

        for batch in self.loader:
            batch = batch.to(self.device, non_blocking=True)
            logits, _ = self.model(batch)
            loss = self.loss_fn(logits, batch[:, -1])

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            self.meter.update(batch.numel())
            self.step += 1

            if self.step % self.cfg.log_interval == 0:
                tps = self.meter.rate/1e3
                print(f"step {self.step:06d}  loss {loss.item():.4f}  {tps:.1f} Ktok/s")

            if self.step >= self.cfg.max_steps:
                break
            if self.meter.counter >= tokens_target:
                break

        elapsed = time.time() - start
        print(f"finished {self.step} steps in {elapsed/60:.1f} min")

    # ---------------------------------------------------------------------
    def save(self, path: str):
        save_checkpoint(self.model, self.optim, self.step, path)
