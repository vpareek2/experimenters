from __future__ import annotations
"""Unified dataset loader – converts a :class:`DataConfig` into a
``torch.utils.data.IterableDataset`` ready for the Trainer.

Supported ``src`` prefixes
--------------------------
* ``hf:<dataset_name>``   – HuggingFace Hub datasets (streaming if requested)
* ``txt:<glob>``          – local plaintext files (one doc per line)
* *callable*              – a Python function that returns an IterableDataset
"""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import IterableDataset

from experimenters.data.schema import DataConfig

try:
    import datasets as hf_datasets  # type: ignore
    import transformers             # type: ignore
except ImportError:  # soft deps; only required when used
    hf_datasets = None  # type: ignore
    transformers = None  # type: ignore

# ---------------------------------------------------------------------------
#  HF streaming dataset
# ---------------------------------------------------------------------------

class _HFIterDataset(IterableDataset):
    def __init__(self, dconf: DataConfig, seq_len: int):
        if hf_datasets is None:
            raise ImportError("`datasets` package required for HF sources")
        name = dconf.src.split(":", 1)[1]
        self.ds = hf_datasets.load_dataset(name, split="train", streaming=dconf.streaming)

        # tokenizer
        tok = dconf.tokenizer
        if isinstance(tok, str):
            if transformers is None:
                raise ImportError("`transformers` needed for tokenizer")
            tok = transformers.AutoTokenizer.from_pretrained(tok, use_fast=True)
            tok.pad_token = tok.eos_token
        self.tokenizer = tok  # type: ignore

        self.seq_len = seq_len
        self.field = dconf.text_field or "text"

    def __iter__(self):
        cache: list[int] = []
        for record in self.ds:  # type: ignore
            txt = record[self.field]
            cache.extend(self.tokenizer.encode(txt))  # type: ignore[attr-defined]
            while len(cache) >= self.seq_len:
                yield torch.tensor(cache[: self.seq_len], dtype=torch.long)
                cache = cache[self.seq_len :]

# ---------------------------------------------------------------------------
#  Local text files dataset
# ---------------------------------------------------------------------------

class _TxtIterDataset(IterableDataset):
    def __init__(self, dconf: DataConfig, seq_len: int):
        paths = list(Path(dconf.src[4:]).expanduser().glob("**/*"))
        if not paths:
            raise FileNotFoundError(f"No files match pattern {dconf.src[4:]}")

        tok = dconf.tokenizer
        if isinstance(tok, str):
            if transformers is None:
                raise ImportError("`transformers` needed for tokenizer")
            tok = transformers.AutoTokenizer.from_pretrained(tok, use_fast=True)
            tok.pad_token = tok.eos_token
        self.tokenizer = tok  # type: ignore

        self.paths = paths
        self.seq_len = seq_len

    def _lines(self):
        for p in self.paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

    def __iter__(self):
        cache: list[int] = []
        for txt in self._lines():
            cache.extend(self.tokenizer.encode(txt))  # type: ignore[attr-defined]
            while len(cache) >= self.seq_len:
                yield torch.tensor(cache[: self.seq_len], dtype=torch.long)
                cache = cache[self.seq_len :]

# ---------------------------------------------------------------------------
#  Public loader
# ---------------------------------------------------------------------------

def load_dataset(dconf: DataConfig, model_cfg) -> IterableDataset:  # type: ignore
    """Return iterable dataset according to *dconf* and *model_cfg* (for seq_len)."""
    seq_len = getattr(model_cfg, "seq_len", 256)

    # custom factory
    if callable(dconf.src):
        return dconf.src(model_cfg)  # type: ignore[arg-type]

    if dconf.src.startswith("hf:"):
        return _HFIterDataset(dconf, seq_len)

    if dconf.src.startswith("txt:"):
        return _TxtIterDataset(dconf, seq_len)

    raise ValueError(f"Unrecognised data src: {dconf.src}")
