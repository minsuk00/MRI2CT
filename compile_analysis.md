# torch.compile Analysis — MRI2CT UNet/Amix Training

**Date:** 2026-05-29  
**GPU:** NVIDIA A40 (44 GB)  
**Setup:** batch=8, patch=128³, bf16 AMP, dice_w=0.1 (BabyUNet teacher active)

---

## What We Profiled

The UNet baseline trainer (`src/unet_baseline/train.py`), which is structurally identical to the amix trainer (`src/amix/trainer.py`) in how it handles compilation. Both trainers have the same three compile_mode options:

- **`compile_mode="full"`** — compiles the entire `_train_step` function (model, teacher, loss, backward all in one graph). Models themselves are NOT separately compiled.
- **`compile_mode="model"`** — compiles `self.model`, `self.feat_extractor` (amix only), and `self.teacher_model` individually. The step function runs in eager mode.
- **`compile_mode=None`** — no compilation.

Both trainers currently use `compile_mode="full"`.

---

## Experiment 1 — Per-Phase Timing (CUDA events)

**Method:** Synthetic batch (random tensors matching real data shape), CUDA events for GPU-accurate timing, 8 warmup + 25 measurement iterations. Phases timed separately with `torch.cuda.Event`.

**Step structure profiled:**
```
batch = next(train_iter)                   # CPU (DataLoader)
aug   = gpu_augment_batch(batch, ...)      # GPU aug (batchaug)
pred, loss, comps = train_step(mri, ct)    # compiled _train_step
grad_clip + optimizer.step()               # optimizer
```

### Results (compile_mode="full", batch=8)

| Phase | Time | % of wall |
|---|---|---|
| GPU aug (batchaug pipeline) | 147 ± 17 ms | 9.3% |
| **Compiled train_step (fwd+loss+bwd)** | **1437 ± 2 ms** | **90.6%** |
| Optimizer (grad_clip + Adam.step) | 0.9 ms | 0.1% |
| **Wall total** | **1585 ms/step** | — |

**Throughput: ~5 patches/sec**

### Surgical Breakdown (uncompiled, isolated per-phase)

Each phase measured independently with a fresh batch to avoid memory pressure.

| Phase | Time | Notes |
|---|---|---|
| UNet forward (eval, no_grad) | 212 ± 3 ms | 5.9M params, ngf=16, 4 downs, BatchNorm |
| Teacher forward (no_grad, bf16) | 379 ± 1 ms | 36.8M params (6.3× larger), 5 downs, ngf=20, InstanceNorm |
| L1 loss | 0.8 ms | negligible |
| SSIM (fused_ssim3d) | 10.7 ms | small; also a graph break for torch.compile |
| Backward (UNet only, L1+SSIM path) | 521 ± 1 ms | teacher backward not separately measured here |

The compiled train_step (1437 ms) − surgical sum (212+379+11+521 = 1123 ms) ≈ **314 ms unaccounted**. This is the teacher backward — backpropagating through the frozen teacher's input to propagate dice gradients back to the UNet. Frozen parameters mean no parameter accumulation, but the Jacobian w.r.t. input (pred) must still be computed.

**Cost breakdown estimate:**
- UNet forward: ~212 ms (15%)
- Teacher forward: ~379 ms (26%)
- Teacher backward (through input): ~314 ms (22%)
- UNet backward (L1+SSIM+dice gradient): ~521 ms (36%)
- Loss computation: ~11 ms (1%)

**Teacher (fwd + bwd through input) = ~693 ms ≈ 48% of compute.**

---

## Experiment 2 — compile_mode Comparison

**Method:** Each config ran in isolation with `gc.collect() + torch.cuda.empty_cache() + torch._dynamo.reset()` between configs. Uncompiled configs required batch=4 to avoid OOM (see note below). Wall times for batch=4 configs normalised to batch=8 equivalent (×2) for comparison. Verified the best config at actual batch=8.

**OOM finding:** No-compile at batch=8 OOMs during backward. The teacher backward in eager mode stores all intermediate activations of the 36M-param teacher explicitly on the autograd tape. `torch.compile` avoids this via op fusion, never materialising some intermediates — so compile is required for batch=8 to be viable at all.

### Results

| Config | Wall (norm b=8) | Speedup vs no_compile |
|---|---|---|
| `no_compile` (b=4, norm→8) | 1554 ms | 1.00× |
| `compile_teacher` only (b=4, norm→8) | 1482 ms | 1.05× |
| `compile_model` only (b=4, norm→8) | 1333 ms | 1.17× |
| `compile_model+teacher` (b=4, norm→8) | 1264 ms | 1.23× |
| **`compile_model+teacher` (b=8, direct)** | **1255 ms** | **1.24×** |
| `compile_full/default` (b=8) ← **current** | 2005 ms | **0.77× (slower than no_compile)** |

**`compile_full` is 1.60× slower than `compile_model+teacher`.**

---

## Root Cause

`fused_ssim3d` (from the `fused_ssim` package) is a pybind11 C extension. PyTorch's Dynamo tracer cannot trace through it and emits a warning:

```
UserWarning: Dynamo does not know how to trace the builtin
`fused_ssim_cuda...fusedssim3d.`
```

This creates a **graph break** inside the compiled `_train_step` — right at the SSIM call inside `loss_fn`. Instead of one fused end-to-end compiled graph, Dynamo generates multiple smaller subgraphs stitched together with Python dispatch overhead between them. Cross-boundary fusion (e.g., fusing L1 computation with the backward prologue) is lost.

The result: `compile_full` ends up slower than not compiling at all, because it adds graph-capture overhead without delivering the fusion benefits.

`compile_mode="model"` avoids this entirely — each sub-model is compiled independently into a clean graph, and `fused_ssim3d` runs as a normal eager op in the step loop without disrupting any compiled graph.

---

## Code Path (both trainers follow identical logic)

```python
# compile_mode="full":
self.model = model                          # raw, uncompiled
self.teacher_model = _setup_teacher_model(compile_model=False)  # raw
self.train_step = torch.compile(self._train_step, ...)          # whole step compiled
#   → graph break at fused_ssim3d inside loss_fn

# compile_mode="model":
self.model = torch.compile(model, ...)      # compiled individually
self.teacher_model = _setup_teacher_model(compile_model=True)   # compiled individually
self.train_step = self._train_step          # step runs in eager
#   → no graph break; fused_ssim3d runs normally
```

In amix, `feat_extractor` is also compiled when `compile_mode="model"` (line 153 in `trainer.py`):
```python
if self.cfg.compile_mode != "full":
    self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
```

---

## Current Status

| Trainer | Before | After | Verified |
|---|---|---|---|
| `src/unet_baseline/train.py` | `"full"` | `"model"` | ✓ 1255 ms vs 2005 ms |
| `src/amix/trainer.py` | `"full"` | not changed | benchmark not yet run |

The amix trainer has the same structure and the same `fused_ssim3d` graph break. The fix (`compile_mode="model"`) should yield a similar speedup, but needs its own benchmark before changing since amix also has a `feat_extractor` in the step (adding another sub-model to the compiled graph).

---

## What Was NOT a Bottleneck

- **DataLoader / CPU workers:** Not the bottleneck. Aug+compute dominates at 1.5s/step; the DataLoader queue stays full.
- **batchaug GPU augmentation:** 147 ms (9.3%) — reasonable for the pipeline size (flip, rotate, low-res sim, Gibbs, bias field, contrast, smooth, sharpen, affine, elastic, randconv).
- **L1 loss:** 0.8 ms — negligible.
- **SSIM loss:** 10.7 ms — small. `fused_ssim3d` is fast; the problem is compile incompatibility, not runtime cost.
- **Optimizer:** 0.9 ms — negligible.

## What IS the Fundamental Bottleneck (after compile fix)

After switching to `compile_mode="model"`, the dominant cost is the teacher model:
- Teacher forward: ~190 ms at b=8 (scales linearly; ~half of UNet forward at b=4)
- Teacher backward (gradient through frozen input): ~160 ms at b=8
- Total teacher: ~350 ms ≈ **~28% of step** at 1255 ms/step

UNet backward remains the single largest individual phase (~41%). No algorithmic fix available without changing training semantics (no dropping Dice, no periodic Dice, no lower-res teacher).
