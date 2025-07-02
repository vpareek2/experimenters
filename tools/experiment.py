# tools/experiment.py
"""Command-line entry point for quick experiments.

Usage
-----
$ python -m tools.experiment \
      --cfg configs.moe_256  \
      --set n_layers=12 attn.latent_dim=256

The script expects a Python module or file that exposes a top-level ``cfg``
object (typed dataclass from *experimenters.config.schema*).  Dot-list
overrides let you tweak any field without editing the file.
"""
from __future__ import annotations

import argparse

from experimenters.config.io import load_py_cfg, apply_dotlist_overrides
from experimenters.model.transformer import build_transformer
from experimenters.trainer import Trainer
from experimenters.registry import ATTN, FFN

# -----------------------------------------------------------------------------
# 1. CLI parsing
# -----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Experimenters – rapid LLM sandbox")
    p.add_argument(
        "--cfg",
        required=True,
        help="Python file or dotted module path containing a top‑level 'cfg' object.",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Override fields in the config (dot‑notation).",
    )
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# 2. Main entry
# -----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # 2.1 Load + mutate config
    cfg = load_py_cfg(args.cfg)
    cfg = apply_dotlist_overrides(cfg, args.set)

    # 2.2 Lookup component classes via registry
    attn_cls, _ = ATTN[cfg.attn_type]
    ffn_cls, _ = FFN[cfg.ffn_type]

    # 2.3 Build model + trainer
    model = build_transformer(cfg, attn_cls=attn_cls, ffn_cls=ffn_cls)
    trainer = Trainer(model, cfg)

    # 2.4 Go!
    trainer.train()


if __name__ == "__main__":
    main()
