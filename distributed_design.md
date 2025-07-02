# Parallelism Roadmap — Experimenters

> **Goal:** let a user scale from laptop demos to 30-40 B-param research models *without* rewriting training scripts.
> **Scope:** 1 – 8 GPUs on one host; multiple hosts out-of-scope for v0.x.

---

## 0  Baseline always-on

| Technique             | Status | Notes                                |
| --------------------- | ------ | ------------------------------------ |
| Gradient accumulation | **on** | `cfg.grad_accum_steps`, default = 1  |
| AMP / BF16 autocast   | **on** | CPU / MPS fallback to FP32           |
| Grad-clip             | **on** | `cfg.grad_clip_norm` (None disables) |

No comms; works on CPU/MPS too.

---

## 1  Config knobs

```python
@dataclass
class ModelConfig:
    # ----- topology knobs -----
    tp_size:      int | str = "auto"   # tensor-parallel group width
    pp_stages:    int | str = "auto"   # pipeline stages
    dp_size:      int | str = "auto"   # data-parallel replicas
    # derived automatically if "auto"
```

### Default auto-heuristic (`experimenters.parallel.init_dist`)

```text
if  model_params <= 2B and seq_len <= 2k:  TP = PP = 1
elif model_params <= 7B:                   TP = 2, PP = 1
else:                                      TP = 4, PP = 2

DP = world_size // (TP * PP)
```

*Safety margin* checks activation-peak; user can always override with integers.

---

## 2  Runtime initialisation

```python
from experimenters.parallel import init_dist, get_tp_pg, get_pp_pg
init_dist(tp_size, pp_stages)      # sets torch.distributed groups
```

Creates three communicators:

| Group      | Span                                           | Used for                 |
| ---------- | ---------------------------------------------- | ------------------------ |
| **dp\_pg** | all ranks sharing *identical* weights          | grad all-reduce          |
| **tp\_pg** | ranks inside same pipeline stage               | QKV/MLP shard all-reduce |
| **pp\_pg** | send/recv between stages (rank, rank+tp\_size) | activations / grad-push  |

---

## 3  Code-path adjustments

### 3.1 Tensor-parallel aware layers

* `tensor_parallel.Linear` with `mode="row"` / `"col"`
* `tensor_parallel.MHA` (wraps Flash-Attn when available)
* Choose sliced or regular kernel based on `tp_size > 1`.

### 3.2 Pipeline wrapper (optional)

```python
from experimenters.parallel.pipeline import Stage

class TransformerStage(nn.Module):
    def __init__(self, layers, stage_id, total_stages):
        ...
    def forward(self, x):
        x = send_recv_prev(x)            # if stage_id > 0
        for blk in self.layers: x = blk(x)
        x = send_recv_next(x)            # if stage_id < last
        return x
```

`xp.build_transformer()` instantiates 1 or 2 stages depending on `pp_stages`.

### 3.3 Trainer

* 1F1B micro-batch schedule when `pp_stages>1`.
* `DataLoader` world-size set to `dp_size`; each DP replica sees a unique data shard.

---

## 4  Checkpoint & resume

* **TP shards:** save per-rank weight slice with `.tp{rank}.pt` suffix.
* **PP stages:** include `stage_id` in path.
* **DP:** identical states; one replica writes, others skip I/O.

Loader gathers slices if `tp_size` differs between training and finetune.

---

## 5  Callback visibility

`trainer.metrics` already exposes:

* `dp_size`, `tp_size`, `pp_stages`
* FLOPs/sec, tokens/sec (computed once topology is known)

Callbacks (PrintLoss, TBLogger) automatically print/log them.

---

## 6  Testing matrix (CI)

| GPU env       | Expected topology      | Smoke test                       |
| ------------- | ---------------------- | -------------------------------- |
| CPU or MPS    | DP = 1, TP = PP = 1    | unit tests                       |
| CUDA (2 GPUs) | TP = 1, DP = 2         | `examples/gpt2_tiny.py` 10 steps |
| CUDA (4 GPUs) | TP = 2, DP = 2         | synthetic 2B model 5 steps       |
| CUDA (8 GPUs) | TP = 4, PP = 2, DP = 1 | build-only (skip train)          |

---

## 7  Future hooks

* **Expert parallel** will reuse `dp_pg` for all-to-all dispatch; only gate runs across DP replicas.
* **FP8 kernels** live inside TP-aware matmul wrappers (`linear_fp8.py`); no change to Trainer.
* **FSDP** can replace DDP by plugging an `xp.parallel.enable_fsdp(model)` before training loop.

---

### TL;DR

* Default = CPU-friendly single-GPU, single-process.
* Auto-detected TP/PP only when params & activations demand it.
* Users override with integers or `1` to disable.
* One `init_dist()` call sets up all communicators; Trainer and blocks query helpers (`get_tp_pg`, etc.).

This keeps the baseline intuitive while unlocking memory headroom and throughput on 8×H100 rigs when you flip a couple of config knobs.
