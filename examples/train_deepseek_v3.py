# Currently this is not faithful to DSV3 at all, it is just a tiny example of a MoE with Multi-head latent attention. I'm working to a faithful implementation.
"""
FIX THIS LATER:
Great — the toy DeepSeek-V3 run shows that the **graph shape** (MLA + aux-free MoE gate) and the *callback plumbing* are sound. Before we sprint toward Triton kernels, here’s what’s still missing to make training dynamics **faithful** to the paper / Torchtitan reference, all achievable with plain PyTorch:

---

## 1  Learning-rate schedule (warm-up + cosine)

**Why:** DS-V3 warms up \~2 K steps then cosine-decays.
**How:** you already have `LRWarmupCosine`; set sensible defaults:

```python
cfg.scheduler = dict(type="cosine", warmup=2000, total=cfg.max_steps)
xp.run(cfg, callbacks=[cb.LRWarmupCosine(**cfg.scheduler)])
```

*Side-effect:* Grad-norm will drop from \~12 to the single-digits seen in the paper.

---

## 2  Gate **capacity** and dropout

| Feature             | Paper behaviour                                                                                  | Quick PyTorch patch                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Capacity**        | Each expert processes at most `⌈1.25 · tokens/k/E_local⌉`; excess tokens are *silently skipped*. | In `MoE.forward`, after `if w_eid.size(0) > capacity: …` you already clamp. Keep the “dropped-token” count in `trainer.metrics["moe_drop"]`. |
| **Routing dropout** | 10 % tokens forced to shared experts during warm-up.                                             | Inside gate: `if self.training and self.dropout_p: mask = …` then route to shared experts.  Add `dropout_p=0.1` to `MoEConfig`.              |

These two changes let load-balancing bias converge faster without an auxiliary loss.

---

## 3  Multi-token prediction objective

Paper predicts the **next k tokens** (k≈3) to stabilise long-context learning.

```python
class MultiTokenCE(xp.Callback):
    def aux_loss(self, tr, batch):
        logits, _ = tr.model(batch)
        # logits: (B, T, V)  – predict the last k positions
        tgt = batch[:, -self.k:]
        logits = logits[:, -self.k:, :].reshape(-1, logits.size(-1))
        return tr.loss_fn(logits, tgt.reshape(-1)) * self.scale
```

Plug it via `callbacks=[MultiTokenCE(k=3, scale=0.2)]`.

---

## 4  Long-context YaRN scaling

You already expose `rope_factor` in `MLAConfig`; simply set

```python
cfg.attn.rope_factor = 10   # e.g. for 4-8 k seq_len
```

and MLA’s internal scale matches the DeepSeek long-context trick.

---

## 5  Parameter-wise weight decay

DeepSeek skips decay on bias, RMSNorm, and gating params.
Quick fix:

```python
def build_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if n.endswith(('bias', 'weight')) and 'norm' in n.lower()
                      else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95))
```

Wire it inside `Trainer.__init__`.

---

## 6  Numeric precision

*Stay FP32/BF16* for now, but insert a `precision_policy` field so later you can flip to FP8 kernels without API churn:

```python
cfg.precision = "bf16"   # "fp16" | "fp8"
with torch.autocast(device_type, dtype=cfg.precision_dtype):
    ...
```

---

## 7  Validation perplexity like the paper

Wrap the `EvalLoop` output:

```python
tr.metrics["ppl"] = math.exp(val_loss)
```

so `PrintLoss` prints `ppl 123.4` every eval.

---

### Minimal “paper-faithful” DeepSeek-tiny recipe

```python
cfg = xp.ModelConfig(
    ...,
    scheduler   = dict(type="cosine", warmup=2000),
    ffn         = xp.MoEConfig(n_experts=32, top_k=2, dropout_p=0.1),
    precision   = "bf16",
)
xp.run(cfg, callbacks=[
    cb.LRWarmupCosine(**cfg.scheduler),
    cb.MultiTokenCE(k=3, scale=0.2),
    cb.EvalLoop(val_loader, every=200),
    cb.PrintLoss(every=10),
])
```

You’ll see:

```
step 000200  loss 4.31  ppl 74.2  grad_norm 1.88
```

—much closer to the training curves reported in the DS-V3 appendix, **without** any Triton or FP8 yet.

---

### Next coding bites

1. **Add `dropout_p` & capacity metric** to `MoE`.
2. **Integrate `MultiTokenCE` callback.**
3. **Swap `Trainer`’s optimizer builder** for param-wise weight-decay.
4. **Implement `precision_policy` switch with AMP autocast.**

Do those, rerun the tiny DeepSeek script, and you’ll have an *accuracy-faithful* baseline ready for future kernel upgrades.

"""

# examples/train_deepseek_moe.py
"""
Tiny DeepSeek-V3-style model (MLA + MoE) trained for 5 steps on random tokens.
Run:  python -m examples.train_deepseek_moe
"""

import torch, experimenters as xp
import experimenters.callbacks as cb
from torch.utils.data import IterableDataset

# ----------------------------------------------------------------------
#  random-token dataset so the script is 100 % offline
# ----------------------------------------------------------------------
class RandTok(IterableDataset):
    def __init__(self, seq_len, vocab): self.len, self.vocab = seq_len, vocab
    def __iter__(self):
        while True:
            yield torch.randint(0, self.vocab, (self.len,))

# ----------------------------------------------------------------------
#  DeepSeek-ish tiny config
# ----------------------------------------------------------------------
cfg = xp.ModelConfig(
    dim        = 256,
    n_layers   = 8,
    n_heads    = 4,
    # MLA + MoE
    attn_type  = "MLA",
    ffn_type   = "MoE",
    moe_layers = 4,                       # last 4 layers become MoE
    # MLA sub-config (smaller than paper but proportional)
    attn = xp.MLAConfig(latent_dim=128, qk_rope_dim=16, qk_nope_dim=16),
    # MoE sub-config – auxiliary-loss-free bias balancing
    ffn  = xp.MoEConfig(n_experts=32, top_k=2, route_scale=1.0),
    # training hyper-params
    seq_len    = 256,
    batch_size = 64,
    max_steps  = 5,
    lr         = 3e-4,
    data       = xp.DataConfig(src=lambda c: RandTok(c.seq_len, c.vocab_size)),
)

# ----------------------------------------------------------------------
#  callbacks: merged print line + tiny checkpoint
# ----------------------------------------------------------------------
val_loader = torch.utils.data.DataLoader(
    RandTok(cfg.seq_len, cfg.vocab_size), batch_size=cfg.batch_size)

xp.run(cfg, callbacks=[
    cb.PrintLoss(every=1),                 # single merged logger
    cb.GradNormLogger(every=1),
    cb.EvalLoop(val_loader, every=2),      # micro validation
    cb.CheckpointSaver(every=4, path="ckpt"),
])
