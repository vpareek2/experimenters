# tools/experiment.py
import argparse
import importlib.util
import sys

from experimenters.config import load_cfg
from experimenters.model.transformer import build_transformer
from experimenters.trainer import Trainer
from experimenters.registry import ATTN, FFN               # auto-lookup

def dynamic_import(py_file: str):
    spec = importlib.util.spec_from_file_location("user_model", py_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = mod
    spec.loader.exec_module(mod)
    return mod

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model",  required=False, help="python file defining build_model(cfg)")
    args = p.parse_args(argv)

    cfg = load_cfg(args.config)

    if args.model:
        mod = dynamic_import(args.model)
        model = mod.build_model(cfg)        # user provided
    else:
        attn_cls, _ = ATTN[cfg.attn_type]
        ffn_cls,  _ = FFN [cfg.ffn_type]
        model = build_transformer(cfg, attn_cls=attn_cls, ffn_cls=ffn_cls)

    trainer = Trainer(model, cfg)
    trainer.train()

if __name__ == "__main__":
    main()
