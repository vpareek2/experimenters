# experimenters/config/__init__.py
import yaml, dacite
from pathlib import Path
from experimenters.config.model import TransformerCfg    # re-export

def load_cfg(path: str | Path) -> TransformerCfg:
    data = yaml.safe_load(Path(path).read_text())
    return dacite.from_dict(TransformerCfg, data)
