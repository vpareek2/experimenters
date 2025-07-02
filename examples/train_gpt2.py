import experimenters as xp
from experimenters import callbacks
import torch
from torch.utils.data import IterableDataset, DataLoader

# ---------------------------------------------------------------------------
# 1. Random-token train + val datasets (offline)
# ---------------------------------------------------------------------------
class RandTok(IterableDataset):
    def __init__(self, seq_len: int, vocab: int):
        self.seq_len, self.vocab = seq_len, vocab
    def __iter__(self):
        while True:
            yield torch.randint(0, self.vocab, (self.seq_len,))

# cfg helper for lambda
train_factory = lambda c: RandTok(c.seq_len, c.vocab_size)
val_loader    = DataLoader(RandTok(128, 50_257), batch_size=32)

# ---------------------------------------------------------------------------
# 2. Model config (MHA + GELU FFN)
# ---------------------------------------------------------------------------
cfg = xp.ModelConfig(
    dim=128, n_layers=4, n_heads=4,
    attn_type="MHA", ffn_type="GELU",
    seq_len=128, batch_size=32, max_steps=5,
    data=xp.DataConfig(src=train_factory),
    lr=3e-4,
)

# ---------------------------------------------------------------------------
# 3. Callback bundle
# ---------------------------------------------------------------------------
xp.run(cfg, callbacks=[
    callbacks.PrintLoss(every=1),
    callbacks.GradNormLogger(every=1),
    callbacks.EvalLoop(val_loader, every=2),
    callbacks.CheckpointSaver(every=100),
])


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    xp.run(cfg, callbacks=callbacks)
