# System-Design Overview for **experiments** (v 0.1)

> A lightweight, plug-and-play transformer sandbox where researchers can drag-and-drop building blocks and run quick training loops without heavyweight dependencies.

---

## 1 . Project Goals & Design Principles

| Goal                               | Principle                              | Practical Consequence                                     |
| ---------------------------------- | -------------------------------------- | --------------------------------------------------------- |
| **1. One-liner imports**           | *Flat public namespace*                | `from experiments.attn import MLA, SDPA`                  |
| **2. Rapid experimentation**       | *Composable builder + minimal trainer* | Swap any block via a config string or CLI flag.           |
| **3. Zero-surprise layout**        | *One component per file*               | Easier code search & PR reviews.                          |
| **4. No hard deps beyond PyTorch** | *Extras tags for speed-ups*            | `pip install experiments[fast]` pulls Triton, Flash-Attn. |
| **5. Gradual extensibility**       | *Optional registries & callbacks*      | You can add features without moving files again.          |

---

## 2 . Directory & Package Layout

```
experiments/                 ← installable root (set in pyproject.toml)
│
├── __init__.py              ← re-exports public classes & helper funcs
│                              (__all__ lists keep IDE autocompletion clean)
│
├── attn/                    ← Attention mechanisms
│   ├── __init__.py          ← from .mla import MLA …      (public names)
│   ├── mla.py               ← Multi-Head Latent Attention
│   ├── mha.py               ← Vanilla Multi-Head Attention
│   ├── sdpa.py              ← torch-native Scaled Dot-Product Attention
│   └── native_sparse.py     ← DeepSeek-style sparse kernel
│
├── ffn/                     ← Feed-forward variants
│   ├── __init__.py
│   ├── swiglu.py
│   ├── gated_mlp.py
│   └── moe.py
│
├── norm/                    ← Normalisation layers
│   ├── __init__.py
│   ├── rmsnorm.py
│   └── layernorm.py
│
├── embed/                   ← Embeddings
│   ├── __init__.py
│   ├── token.py             ← TokenEmbedding
│   └── parallel.py          ← Row-/column-sharded embeddings
│
├── model/                   ← High-level model builders
│   ├── __init__.py
│   └── transformer.py       ← build_transformer(cfg, *, attn_cls, ffn_cls…)
│
├── trainer/                 ← HF-style but minimal training utilities
│   ├── __init__.py
│   └── mini_trainer.py      ← MiniTrainer
│
├── utils/                   ← Small helpers, no deps
│   ├── __init__.py
│   ├── dist.py              ← get_world_size(), get_rank()
│   ├── metrics.py           ← ThroughputMeter, AverageMeter
│   └── checkpoint.py        ← save_ckpt(), load_ckpt()
│
└── registry.py (optional)   ← String→class phone-book (see §6)
configs/                     ← YAML configs (tiny.yaml, 1b.yaml…)
docs/                        ← MkDocs site (overview + API)
examples/                    ← Quick-start notebooks / scripts
tools/                       ← Thin CLIs (train.py, eval.py, convert_ckpt.py)
tests/                       ← PyTest unit + smoke tests
```

---

## 3 . Core Modules

### 3.1 `model/transformer.py`

```python
def build_transformer(cfg,
                      *,
                      attn_cls,
                      ffn_cls,
                      norm_cls,
                      embed_cls):
    """
    Returns nn.Module wired with the provided component classes.
    cfg is a dataclass with dim, n_layers, n_heads, etc.
    """
```

*Stateless*; accepts any classes that follow the expected init/forward signature.
Future: add `compile=True` flag that runs `torch.compile` if available.

### 3.2 `trainer/mini_trainer.py`

Minimal yet HF-flavoured:

```python
trainer = MiniTrainer(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader=None,
    epochs=3,
    device="auto",
    callbacks=[WandBLogger(), LRDecay()],
    ckpt_dir="ckpt/run1",
)
trainer.train()
```

Supported v0.1 features

| Feature                                    | Implementation note                                              |
| ------------------------------------------ | ---------------------------------------------------------------- |
| Single GPU (`device="cuda:0"` or `"cpu"`)  | default                                                          |
| Automatic AMP (fp16 / bf16)                | `torch.autocast` param                                           |
| Simple LR scheduler                        | pass any `torch.optim.lr_scheduler` object                       |
| Callbacks (`on_batch_end`, `on_epoch_end`) | list of callables                                                |
| Checkpoint save/resume                     | via `utils.checkpoint`                                           |
| Optional DDP                               | later: `auto_ddp=True` injects wrapper around model & DataLoader |

---

## 4 . Public API Surface (import paths)

```python
from experiments.attn import MLA, SDPA
from experiments.ffn  import SwiGLU
from experiments.norm import RMSNorm
from experiments.embed import TokenEmbedding

from experiments.model.transformer import build_transformer
from experiments.trainer import MiniTrainer
```

*Every public symbol lives exactly one import-level below `experiments.*`.*

---

## 5 . CLI Entry Points

Declared in **`pyproject.toml`**:

```toml
[project.scripts]
exp-train = "experiments.tools.train:main"
exp-eval  = "experiments.tools.eval:main"
```

CLI snippet:

```bash
exp-train \
  --attn MLA \
  --ffn SwiGLU \
  --layers 6 \
  --batch 128 \
  --seq-len 256 \
  --epochs 3 \
  --save-dir ckpt/mla_demo
```

The wrapper script loads YAML defaults from `configs/tiny.yaml`, overrides with CLI args, instantiates classes via `importlib` (or registry, §6), then hands everything to `MiniTrainer`.

---

## 6 . (Opt-In) Registry Design

*Not mandatory for v0.1 but 100 % backward-compatible.*

```python
# registry.py
class Registry(dict):
    def register(self, name):
        def wrap(obj):
            self[name] = obj
            return obj
        return wrap

ATTN = Registry(); FFN = Registry()
```

Each component file decorates itself:

```python
# attn/mla.py
from experiments.registry import ATTN
@ATTN.register("MLA")
class MLA(nn.Module): ...
```

*Why?* Configuration-driven choice:

```python
attn_cls = experiments.registry.ATTN[cfg.attn]   # "MLA" → class
```

If you skip the registry, scripts just `importlib.import_module(f"experiments.attn.{cfg.attn}")`.

---

## 7 . Packaging & Dependencies

```toml
[project]
name = "experiments"
version = "0.1.0"
dependencies = ["torch>=2.1"]

[project.optional-dependencies]
fast = ["flash-attn>=2.5", "triton>=2.2"]
dev  = ["pytest", "ruff", "black", "pre-commit"]

[tool.setuptools.packages.find]
where = ["experiments"]
```

* Pure-Python wheel, no compiled extensions by default.
* Users who want Triton kernels run `pip install experiments[fast]`.

---

## 8 . Typical User Workflow

1. **Install & clone**

   ```bash
   git clone https://github.com/yourname/experiments
   cd experiments
   pip install -e .[dev]
   ```

2. **Run quick sanity train**

   ```bash
   exp-train --attn MLA --ffn SwiGLU --epochs 1
   ```

3. **Notebook tinkering**

   ```python
   from experiments.attn.native_sparse import NativeSparse
   from experiments.ffn.gated_mlp import GatedMLP
   cfg = TinyConfig(layers=8, dim=512)
   model = build_transformer(cfg,
                             attn_cls=NativeSparse,
                             ffn_cls=GatedMLP,
                             norm_cls=RMSNorm,
                             embed_cls=TokenEmbedding)
   ```

4. **Extend**: drop `attn/ring_attention.py`, add `__all__` export, run tests; it’s available immediately.

---

## 9 . Evolution Roadmap (post-0.1)

| Milestone | Adds                                           | Notes                                               |
| --------- | ---------------------------------------------- | --------------------------------------------------- |
| **v0.2**  | DDP, gradient accumulation, dataset streaming  | MiniTrainer grows `auto_ddp`, `iter_dataset` hooks. |
| **v0.3**  | `maybe_compile(model)` shim + Triton RMSNorm   | Enabled only if `[fast]` extras are installed.      |
| **v0.4**  | Registry v2 with metadata (tags, capabilities) | Facilitates AutoML sweeps.                          |
| **v1.0**  | Stable public API + docs site + PyPI           | SemVer: no breaking changes past this point.        |

---

### Take-away

*Use this document as the canonical blueprint*: every new file or feature should fit one of the boxes above, keeping the import paths, build process, and user ergonomics stable.
