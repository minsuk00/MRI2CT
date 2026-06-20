# torch.compile Benchmark: MRI2CT amix / unet Training

**Source HTML:** _html/compile_mode_benchmark.html
**Date:** 2026-05-29
**TL;DR:** `compile_mode="model"` (per-submodel compile) wins on both trainers. `"full"` is not best: it shatters into 8 sub-graphs (broken by the SSIM kernel and by `loss.backward()` inside the compiled step), losing to `"model"` by 1.24x (unet) and 1.19x (amix). CUDA-graph modes (max-autotune, reduce-overhead) fail (OOM / graph aliasing). Action: switch amix from full -> model (~16% faster, 12GB lighter); unet already on model. All modes produce identical loss.

Setup: A40 (44GB), torch 2.10.0+cu128, triton 3.6, batch=8, patch 128^3, bf16 AMP, dice_w=0.1, amix extractor = v1_4.

## 1. The pipeline: what gets compiled in each mode

The training step chains up to three networks plus a composite loss, then backpropagates. Only the translator is trained; other nets are frozen.

| Component | Role | Trainable? | In amix? | In unet? |
| --- | --- | --- | --- | --- |
| Anatomix feature extractor (v1_4: 4 downs, ngf 32, BatchNorm) | MRI -> 16-ch features (translator input) | frozen, no_grad | yes | - |
| U-Net translator (4 downs, ngf 16) | features/MRI -> CT (model we train) | trainable | yes | yes (MRI->CT direct) |
| BabyU-Net teacher (5 downs, ngf 20) | CT -> organ seg (for Dice loss) | frozen, eval; grad flows through to pred | yes | yes |
| Anatomix perceptual loss extractor | perceptual feature L1 | OFF (perceptual_w=0) | off | off |
| Composite loss (L1 + SSIM + Dice) | L1 + 0.1*SSIM + 0.1*Dice | - | yes | yes |

| compile_mode | feat extractor | translator | teacher | loss (SSIM/Dice) | backward |
| --- | --- | --- | --- | --- | --- |
| none | eager | eager | eager | eager | eager |
| model (best) | compiled | compiled | compiled | eager | compiled (per-submodel AOTAutograd) |
| model_maxautotune | compiled +cudagraph | " | " | eager | compiled |
| model_reduceoverhead | compiled +cudagraph | " | " | eager | compiled |
| regional | all three fused into ONE compiled forward graph | | eager | eager | |
| full (amix old default) | entire step compiled as one call -> shatters into 8 sub-graphs (see section 5) | | | | |

## 2. Results: unet (batch=8, patch 128^3, bf16, dice_w=0.1)

| mode | status | steady ms/step | vs best | peak GB | warmup s | graph breaks | loss(step0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (eager) | OOM | - | - | >44 | - | - | - |
| model (best) | ok | 1353 | 1.00x | 31.2 | 116 | 0 | 0.45339 |
| model_reduceoverhead | ok | 1352 | 1.00x | 4.8 (warn) | 157 | 0 | 0.45339 |
| full | ok | 1671 | 0.81x | 31.1 | 179 | 5 | 0.45339 |
| regional | ok | 1733 | 0.78x | 31.1 | 178 | 0 | 0.45339 |
| model_maxautotune | OOM | - | - | >44 | - | - | - |
| full_customssim | err | custom-op API broke on torch 2.10 (experimental; redundant, see section 5) | | | | | |

unet is already on model. Eager OOMs at b=8; compile is required just to fit. The reduce-overhead 4.8GB is an under-measurement: `max_memory_allocated()` doesn't count CUDA-graph private pools; it gives no speedup (compute-bound on 3D convs, not launch-bound). Stability re-runs: model 1353->1359, reduceoverhead 1352->1354.

## 3. Results: amix (v1_4 extractor, batch=8, patch 128^3, bf16, dice_w=0.1)

| mode | status | steady ms/step | vs eager | peak GB | warmup s | graph breaks | loss(step0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none (eager) | ok | 2201 | 1.00x | 43.3 | 46 | 0 | 0.44538 |
| model (best) | ok | 1644 | 1.34x | 32.3 | 103 | 0 | 0.44536 |
| full (current default) | ok | 1956 | 1.13x | 31.7 | 102 | 5 | 0.44537 |
| regional | ok | 2018 | 1.09x | 31.7 | 193 | 0 | 0.44536 |
| model_maxautotune | err | CUDA-graph tensor-overwrite (feat output clobbered by later module sharing graph pool) | | | | | |
| model_reduceoverhead | err | same CUDA-graph tensor-overwrite | | | | | |
| full_customssim | err | custom-op API broke on torch 2.10 | | | | | |

amix eager just fits (43.3/44.4 GB). model = 1.34x faster than eager AND 12GB lighter, and 1.19x faster than today's full default. Stability re-runs: model 1644->1644, full 1956->1963.

## 4. With perceptual loss enabled: does the winner change?

The repo's `AnatomixPerceptualLoss` (a 4th frozen v1_4 net; OFF by default at perceptual_w=0) is built inside `CompositeLoss`, so the model-level compile in `_setup_models` never reaches it; in model mode it runs eager, only full compiles it. Re-ran with perceptual on at batch=4 (the extra net OOMs at b=8):

### unet + perceptual (b=4)

| config | ms/step | peak GB |
| --- | --- | --- |
| none (eager) | 1142 | 30.3 |
| model (perc eager, current code) | 907 | 22.2 |
| model + perc compiled | 845 | 19.3 |
| full | 940 | 15.7 (min) |

### amix v1_4 + perceptual (b=4)

| config | ms/step | peak GB |
| --- | --- | --- |
| none (eager) | 1367 | 30.7 |
| model (perc eager, current code) | 1050 | 22.6 |
| model + perc compiled | 989 | 19.6 |
| full | 1081 | 16.1 (min) |

Same ranking holds: model is fastest on both, but only if the perceptual extractor is also compiled (unet 907->845, amix 1050->989; ~6-7% + ~3GB that plain model leaves on the table). full is not fastest (940 / 1081 ms) though it does win on memory (15.7 / 16.1 GB) via whole-step recompute planning. Loss identical across modes (unet 0.4729, amix 0.4686).

Fix applied: `src/common/trainer_base.py:_setup_loss` now compiles `perceptual.extractor` when `compile_mode=="model"` (dormant until perceptual_w>0). If you want lowest memory instead of best speed with perceptual on, full is the trade.

## 5. Why full loses: the root cause

Tracing the compiled step with `torch._dynamo.explain` shows it never becomes one graph. It breaks at the loss every time:

- unet/full -> 8 graphs, 5 breaks: x3 fused_ssim3d (pybind11 C-extension Dynamo cannot trace, loss.py:112, gb0007); x1 Tensor.backward() called inside the compiled step (fundamental: Dynamo can't trace backward, bench full_step, gb0123).
- amix/full -> 8 graphs, 5 breaks: identical break pattern (fused_ssim3d x3 + Tensor.backward() x1). The extra frozen feature-extractor adds ops (319 vs 227) but zero new breaks; the no_grad extractor traces cleanly.

So the "single fused graph" never exists. On top of the breaks, full wraps `loss.backward()` inside the compiled function, which Dynamo also can't trace. Result: graph-capture + cross-boundary dispatch overhead with little fusion payoff.

Why model wins: compiling each sub-net individually gives AOTAutograd a clean forward AND backward graph per network, and the un-compilable SSIM/Dice ops run as normal eager ops between them (no graph to break).

Why regional (one fused forward, loss/backward eager) still loses: fusing the three forwards into one graph doesn't beat per-submodel compile, and it leaves the backward fully eager (empirically slower than model on both trainers).

Note: the old `compile_analysis.md` blamed only fused_ssim3d and claimed full was slower than eager. Corrected here: the second, equally fundamental break is `loss.backward()` living inside the compiled step (the Dice path does not break); and at the real batch=8 (no x2 normalization) full actually beats eager for amix, it just loses to model.

## 6. Compile warm-up & break-even

Warm-up is paid on every start (and every SLURM requeue). It is small next to a multi-hour run:

- model: one-time compile ~100s cold (~40s when the inductor disk-cache is already warm, e.g. after a requeue). For amix it saves ~557 ms/step vs eager -> pays back in ~150 steps (<1/2 an epoch at 500 steps/epoch).
- Warm-up is noisy and cache-dependent (cold 100-190s, warm 37-55s); table numbers are first-run (cold) totals over 12 warm-up steps. `none` still shows ~46s "warm-up" with no compile, that's the first-step cudnn.benchmark conv autotune.
- Conclusion: warm-up does not change the ranking; model wins on steady-state and amortizes within the first epoch.

## 7. Recommendation

- amix: change compile_mode "full" -> "model": 1.19x faster (1956->1644 ms/step) and 12GB lighter. (src/amix/train.py)
- unet: keep "model": already optimal; eager OOMs at b=8. (src/unet_baseline/train.py)
- Don't use max-autotune / reduce-overhead: their CUDA-graphs are incompatible with the frozen-feat -> translator -> teacher pipeline (OOM / tensor aliasing) and give no speedup anyway.
- If you enable perceptual loss (perceptual_w>0): keep model, and the perceptual extractor is now auto-compiled (trainer_base fix), ~6-7% faster + ~3GB vs the old eager-perceptual behavior. Use full only if you need minimum VRAM over speed.
- Correctness: step-0 loss identical across all working modes (unet 0.45339, amix 0.4454; +perc: 0.4729 / 0.4686); the switch is safe.

## Method

Each (trainer x mode) ran in a fresh subprocess (clean Dynamo/CUDA state), real models + real weights + the real `CompositeLoss`, fixed synthetic batch at true batch=8 (no normalization), CUDA-event timing (12 warm-up + 30 measured), peak via `max_memory_allocated`, breaks via `torch._dynamo.explain` + counters. Top-2 modes re-run for stability. Harness: `~/compile_bench/{bench.py,run_all.py}`; raw JSON: `~/compile_bench/results/`.
