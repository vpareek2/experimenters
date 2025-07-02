from __future__ import annotations
"""Unified dataset loader – turns a :class:`DataConfig` into a
``torch.utils.data.IterableDataset``.

* HF datasets are streamed if ``streaming=True``.
* Local text files are memory‑efficient (buffered read, no full file in RAM).
* Custom callables are executed directly.
"""
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import IterableDataset

from experimenters.data.schema import DataConfig

try:
    import datasets as hf_datasets
    import transformers
except ImportError:  # Soft dependency; raise only if user actually needs it.
    hf_datasets = None  # type: ignore
    transformers = None  # type: ignore

# -----------------------------------------------------------------------------
#  Low‑level helpers
# -----------------------------------------------------------------------------

class _HFIterDataset(IterableDataset):
    def __init__(self, dconf: DataConfig, seq_len: int):
        if hf_datasets is None:
            raise ImportError("'datasets' package required for HF sources")
        name = dconf.src.split(":", 1)[1]
        self.ds = hf_datasets.load_dataset(name, split="train", streaming=dconf.streaming)
        tok = dconf.tokenizer
        if isinstance(tok, str):
            if transformers is None:
                raise ImportError("'transformers' needed for tokenizer")
            tok = transformers.AutoTokenizer.from_pretrained(tok, use_fast=True)
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self.seq_len = seq_len
        self.field = dconf.text_field or "text"

    def __iter__(self):
        cache: list[int] = []
        for record in self.ds:
            txt = record[self.field]
            cache.extend(self.tokenizer.encode(txt))
            while len(cache) >= self.seq_len:
                yield torch.tensor(cache[: self.seq_len], dtype=torch.long)
                cache = cache[self.seq_len :]


class _TxtIterDataset(IterableDataset):
    def __init__(self, dconf: DataConfig, seq_len: int):
        paths = list(Path(dconf.src[4:]).expanduser().glob("*"))  # after "txt:"
        tok = dconf.tokenizer
        if isinstance(tok, str):
            if transformers is None:
                raise ImportError("'transformers' needed for tokenizer")
            tok = transformers.AutoTokenizer.from_pretrained(tok, use_fast=True)
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self.seq_len = seq_len
        self.paths = paths

    def _file_iter(self):
        for p in self.paths:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line.strip()

    def __iter__(self):
        cache: list[int] = []
        for txt in self._file_iter():
            cache.extend(self.tokenizer.encode(txt))
            while len(cache) >= self.seq_len:
                yield torch.tensor(cache[: self.seq_len], dtype=torch.long)
                cache = cache[self.seq_len :]


# -----------------------------------------------------------------------------
#  Public entry
# -----------------------------------------------------------------------------

def load_dataset(dconf: DataConfig, model_cfg) -> IterableDataset:  # type: ignore
    """Return an :class:`IterableDataset` according to *dconf*.

    ``model_cfg`` is passed only for convenience (e.g. seq_len).
    """
    seq_len = getattr(model_cfg, "seq_len", 256)

    if callable(dconf.src):
        # src is a custom factory (must return IterableDataset)
        return dconf.src(model_cfg)  # type: ignore

    if dconf.src.startswith("hf:"):
        return _HFIterDataset(dconf, seq_len)

    if dconf.src.startswith("txt:"):
        return _TxtIterDataset(dconf, seq_len)

    raise ValueError(f"Unrecognised data src: {dconf.src}")
