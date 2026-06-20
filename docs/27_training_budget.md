# Training sample budget: MRI2CT baselines

**Source HTML:** _html/training_budget.html
**Date:** undated
**TL;DR:** Reference budget = amix/unet ≈ 3.20M 128³ patches (800 epochs x 500 steps x batch 8), pulled from wandb info/samples_seen, not config defaults. MC-DDPM over-trained (12.0M tiny slabs), MAISI (0.60M) and cWDM (0.32M, failed short of 1.2M target) under-trained in full-volume units. KoalAI trains one model per region at 0.50M patches each, already 0.56-1.11x of amix/unet's per-region share. Caveat: a slab != a patch != a volume, so raw counts across units are not 1:1.

Goal: keep the number of training samples (patch / slab / volume) roughly matched across models, no foreground-voxel normalization, just raw sample counts. Numbers are the actual info/samples_seen from each model's wandb run, not config defaults. The real reference is 3.20M 128³ patches, not the 4.0M the current train.py default (1000 epochs) would give.

## 1. Actual samples seen (from wandb)

One "sample" = one item the network sees per forward pass, in whatever native unit the model uses. samples_seen = steps x batch x patches_per_volume x accum as logged by each trainer.

| Model | wandb id | state | sample unit | unit size | samples seen | vs 3.2M | scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amix | 6hjye9gp | finished | 3D patch | 128³ | 3,200,000 | 1.00x | 1 model, all 5 regions |
| unet | 9xmodnhn | finished | 3D patch | 128³ | 3,200,000 | 1.00x | 1 model, all 5 regions |
| MAISI | 5hprtpwl | finished -> extending | full latent volume | whole vol | 600,000 -> 3,200,000 | 0.19x -> 1.00x | 1 model, all regions |
| cWDM | smg8thkr | failed | full wavelet volume | whole padded vol | 315,500 | 0.10x | 1 model, all regions |
| MC-DDPM | a3g28rez | failed | 3D slab | 128x128x4 | 12,019,208 | 3.76x | 1 model, all regions |
| KoalAI (synth) | per-region | nnU-Net | 3D patch | ~96x160x160 † | 500,000 / region | see §2 | 1 model per region |

amix/unet/maisi/mcddpm read directly from info/samples_seen in the run summary. cWDM logs info/global_step (batch=1, so steps = samples). KoalAI is the fixed nnU-Net schedule 1000 epochs x 250 iters x batch 2 = 500k patches per model (deterministic schedule, not run-verified per region). MC-DDPM and cWDM runs ended failed: MC-DDPM at epoch 3004 of a 7000 target, cWDM at 0.32M of a 1.2M target.

† KoalAI patch size is region-specific (nnU-Net auto-plan): abdomen/thorax 96x160x160, head-neck/brain 128x160x128, pelvis 112x112x192.

## 2. KoalAI per-region share of the 3.2M reference

amix/unet train one model over all 427 training subjects, so each region gets a slice of the 3.2M budget proportional to its subject count. KoalAI trains a separate model per region, so the fair comparison is KoalAI's 0.5M against that per-region share, not the full 3.2M.

| Region | train subj | share of 3.2M | KoalAI / region | ratio |
| --- | --- | --- | --- | --- |
| Abdomen | 65 | 487,000 | 500,000 | 1.03x |
| Brain | 60 | 450,000 | 500,000 | 1.11x |
| Head-Neck | 91 | 682,000 | 500,000 | 0.73x |
| Pelvis | 120 | 899,000 | 500,000 | 0.56x |
| Thorax | 91 | 682,000 | 500,000 | 0.73x |
| avg | 85 | 640,000 | 500,000 | 0.78x |

KoalAI at 0.5M patches/region sits inside 0.56-1.11x of the per-region share, close enough that no change is needed for a "similar samples" claim. To match the share exactly, bump KoalAI epochs from 1000 to share / (250 x 2) ≈ 1280 (avg) or 1364 (thorax / head-neck). Pelvis is the only region where amix/unet meaningfully out-sample KoalAI (1.8x).

## 3. What to change to match 3.2M

| Model | now | target ≈3.2M samples | knob |
| --- | --- | --- | --- |
| MAISI | 0.60M vol | 3.2M vol | configured EPOCHS 1500->8000 in train_maisi_noaug.sh (resume 5hprtpwl, constant LR 1e-4, ~20d GPU) |
| cWDM | 0.32M vol (failed) | 3.2M vol | finish + extend iters to 3.2M (current target was only 1.2M) |
| MC-DDPM | 12.02M slab | 3.2M slab | total_epochs 7000 -> ~1870 (slabs are 1/32 a 128³ patch, so far less over-trained than it looks) |
| KoalAI | 0.5M / region | per-region share (§2) | already ~0.78x avg; optional num_epochs 1000 -> ~1280 |

Caveat (deliberately not normalized): this report counts samples, not voxels. A 128x128x4 MC-DDPM slab holds 1/32 the voxels of a 128³ amix patch, and a full MAISI/cWDM volume holds several patches' worth. Matching raw sample counts across these units does not equalize gradient signal or compute. The honest line is "matched on sample count, units differ," not "matched on data seen."

Source: info/samples_seen / info/global_step from wandb runs 6hjye9gp 9xmodnhn 5hprtpwl smg8thkr a3g28rez (project minsuk-choi/mri2ct); KoalAI from the nnU-Net 1000x250x2 schedule; region subject counts from splits/center_wise_split.txt (427 train).
