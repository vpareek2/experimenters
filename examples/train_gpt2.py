import experimenters as xp
import torch, random
from torch.utils.data import IterableDataset

class RandTok(IterableDataset):
    def __init__(self, seq_len, vocab): self.seq_len, self.vocab = seq_len, vocab
    def __iter__(self):
        while True:
            yield torch.randint(0, self.vocab, (self.seq_len,))

cfg = xp.ModelConfig(
    dim=128, n_layers=4, n_heads=4,
    attn_type="MHA", ffn_type="GELU",     # if you added GELUFFN
    seq_len=128, batch_size=32, max_steps=5,
    data=xp.DataConfig(src=lambda c: RandTok(c.seq_len, c.vocab_size)),
)

model   = xp.build_transformer(cfg)   # now succeeds
xp.Trainer(model, cfg).train()
