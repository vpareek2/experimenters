import experimenters as xp
from experimenters import callbacks
import torch
from torch.utils.data import IterableDataset

# ----- random-token dataset --------------------------------------------------
class RandTok(IterableDataset):
    def __init__(self, seq_len, vocab):
        self.seq_len, self.vocab = seq_len, vocab
    def __iter__(self):
        while True:
            yield torch.randint(0, self.vocab, (self.seq_len,))

# ----- config ----------------------------------------------------------------
cfg = xp.ModelConfig(
    dim=128, n_layers=4, n_heads=4,
    attn_type="MHA", ffn_type="GELU",
    seq_len=128, batch_size=32, max_steps=5,
    data=xp.DataConfig(src=lambda c: RandTok(c.seq_len, c.vocab_size)),
)

# ----- run with callbacks ----------------------------------------------------
xp.run(cfg, callbacks=[
    callbacks.PrintLoss(every=1),
    callbacks.CheckpointSaver(every=100),
])
