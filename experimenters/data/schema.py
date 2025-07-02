from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:  # Only imported during static analysis, no runtime deps
    from torch.utils.data import IterableDataset
    from experimenters.config.schema import ModelConfig

# ---------------------------------------------------------------------------
#  Helper alias – a callable that builds a dataset given the *model cfg*
# ---------------------------------------------------------------------------
DatasetFactory = Callable[["ModelConfig"], "IterableDataset"]

# ---------------------------------------------------------------------------
#  Public DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    src: Union[str, DatasetFactory]
    tokenizer: Union[str, Callable] = "gpt2"
    streaming: bool = True
    text_field: str | None = None

    # trainer hints (optional)
    shuffle: bool = True
    cache_dir: Path | None = None
