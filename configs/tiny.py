from experimenters.config import ModelConf
cfg = ModelConf(
    dim=64, n_layers=2, n_heads=2,
    max_seq_len=32, vocab_size=256,
    batch_size=32, seq_len=32, max_steps=20,
)
