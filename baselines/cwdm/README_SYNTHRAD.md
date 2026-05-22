# cWDM on SynthRAD MR→CT — quick reference

This is the SynthRAD adaptation of [pfriedri/cwdm](https://github.com/pfriedri/cwdm)
(Friedrich et al., 2024). The original cWDM code is left as-untouched-as-possible
and the BraTS path still works via `--dataset=brats --contr=t1n` (etc.). All
SynthRAD additions are guarded by `--dataset=synthrad --contr=ct`.

For the published architecture/diffusion math, read the paper at
`/home/minsukc/MRI2CT/baselines/cWDM.pdf`. This README only covers our integration.

## TL;DR — submit full training

```bash
sbatch /home/minsukc/MRI2CT/sbatch/train_cwdm.sh
```

Edit the variables at the top of that script (`SPLIT_FILE`, `LR_ANNEAL_STEPS`,
`VAL_INTERVAL`, `VAL_SUBJ_ID`, `RESUME_*`, `TAGS`) if needed; defaults match the
paper recipe.

A40 GPU, 96-hour wall clock, 1.2M iterations at lr=1e-5, bs=1.

## Files added / patched

**New (don't touch upstream):**
- `guided_diffusion/synthradloader.py` — MONAI-backed dataset wrapping `src/common/data.py`'s
  `build_data_dicts` + `get_cached_transforms` (same normalization as amix).
- `scripts/validate.py` — full-val-set DDPM evaluator (reduced-step, unpads, masks, saves NIfTI + JSON).
- `/home/minsukc/MRI2CT/sbatch/train_cwdm.sh` — SLURM submission.

**Patched (additive `if dataset == 'synthrad'` branches; BraTS paths untouched):**
- `scripts/train.py` — synthrad dataset branch, wandb init, val-loader/val-diffusion wiring.
- `scripts/sample.py` — single-cond branch, dynamic noise shape, synthrad-aware unpad+mask.
- `guided_diffusion/gaussian_diffusion.py` — `contr == 'ct'` branch in `training_losses`.
- `guided_diffusion/train_util.py` — dataset-aware batch dispatch, wandb log mirror, in-train val hook.
- `run.sh` — `DATASET=synthrad`, `IN_CHANNELS=16`, `--split_file` plumbing.

**Two upstream bug fixes** (necessary for our shapes; left in place for both BraTS and SynthRAD paths):
- `DWT_IDWT/DWT_IDWT_layer.py` — `L1 = max(H, W, D)` instead of `max(H, W)`.
  Symptom: silent matrix truncation when `D > max(H, W)` (BraTS never hits this).
- `guided_diffusion/gaussian_diffusion.py` — `p_sample_loop` defaults `time` to
  `self.num_timesteps` instead of hardcoded 1000.
  Symptom: index-out-of-bounds when sampling from a `SpacedDiffusion(timestep_respacing="ddim50")`.

## Conditioning recipe (single source modality)

For each subject:
```
target  = DWT(CT)                    # 8 wavelet subbands -> (1, 8, D/2, H/2, W/2)
cond    = DWT(MRI)                   # 8 wavelet subbands -> (1, 8, D/2, H/2, W/2)
x_t     = q_sample(target, t)        # add noise in wavelet space
input   = cat([x_t, cond], dim=1)    # (1, 16, D/2, H/2, W/2)
model   = WavUNet(in_channels=16, out_channels=8)
output  = model(input, t)            # predict x0 in wavelet space
loss    = MSE(output, target_dwt)    # uniform over the 8 subbands
```
At inference time: noise → 50 reverse DDPM steps via `SpacedDiffusion(timestep_respacing="ddim50")` → IDWT → unpad → optional body-mask.

## Data normalization

Matches amix exactly (so head-to-head wandb comparisons are apples-to-apples):
- CT: clipped to `[-1024, 1024]` HU → scaled to `[0, 1]`.
- MRI: per-volume `minmax` to `[0, 1]`.
- Padding: each spatial dim padded to ≥128, then to multiple of 32 (required by the
  Haar DWT + 4-level U-Net). Per-subject shapes vary; `bs=1` so collation isn't an issue.
- Pre-pad shape stored as `original_shape` so validation can recover the original geometry.

No data augmentation, matching the published cWDM recipe.

## In-training validation hook

Every `--val_interval` steps the trainer:
1. Loads the fixed subject `--val_subj_id` (default `1THB011`).
2. Runs `--val_ddim_steps` reverse steps (default 50) via a `SpacedDiffusion` over the same model weights.
3. Computes MAE_HU / PSNR / SSIM / grad_diff via `src/common/utils.compute_metrics`
   (same definitions as amix; `hu_range=2048`).
4. Logs `val/mae_hu`, `val/psnr`, `val/ssim`, `val/grad_diff`, `val/sample_image`
   (3-row mid-slice grid: MRI / GT CT / Pred CT / Residual) to wandb.

The hook is intentionally cheap — it's a divergence check, NOT model selection.
For real evaluation use the standalone validator (next section).

## Full validation script

```bash
cd /home/minsukc/MRI2CT/baselines/cwdm

python scripts/validate.py \
  --dataset=synthrad --contr=ct \
  --data_dir=/gpfs/.../SynthRAD/1.5mm_registered_flat_masked \
  --split_file=splits/center_wise_split.txt --split_name=val \
  --model_path=/gpfs/.../wandb_logs/wandb/run-*-<run_id>/files/checkpoints/synthrad_000800000.pt \
  --output_dir=/home/minsukc/MRI2CT/evaluation_results/cwdm_<run_id>_800k/ \
  --ddim_steps=50 --save_nifti=True \
  --wandb_project=mri2ct --wandb_run_name=cwdm_validate_<run_id>_800k \
  --num_channels=64 --num_res_blocks=2 --num_heads=1 \
  --channel_mult=1,2,2,4,4 --diffusion_steps=1000 --noise_schedule=linear \
  --rescale_learned_sigmas=False --rescale_timesteps=False --dims=3 \
  --num_groups=32 --in_channels=16 --out_channels=8 --bottleneck_attention=False \
  --resample_2d=False --additive_skips=False --use_freq=False \
  --predict_xstart=True --learn_sigma=False --use_scale_shift_norm=False \
  --attention_resolutions= --image_size=224
```

Outputs:
- `output_dir/validate_metrics.json` — per-subject and aggregate MAE_HU / PSNR / SSIM / grad_diff (+ body-masked variants).
- `output_dir/<subj>/sample.nii.gz` and `target.nii.gz` at original pre-pad shape, with the right affine.
- WandB run tagged `cwdm`, `validate` — aggregate keys land in `validate/*`.

## Where things land on disk

After `sbatch sbatch/train_cwdm.sh`:
```
/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/        # = WANDB_DIR (same as amix)
└── wandb/
    └── run-<timestamp>-<run_id>/
        └── files/
            ├── output.log            # stdout/stderr
            ├── config.yaml           # all CLI args
            └── checkpoints/
                ├── synthrad_000100000.pt   # model state_dict (~330 MB)
                ├── opt000100000.pt         # AdamW state (~650 MB)
                ├── synthrad_000200000.pt
                └── ...
```
SLURM stdout goes to `/home/minsukc/MRI2CT/slurm_logs/<timestamp>_<job_name>_<job_id>.log`.

Save cadence is fixed (`--save_interval=100000` by default). For a 1.2M-step run
you'll get ~12 checkpoints. This is cWDM's published recipe — different from amix,
which saves on val cadence + tracks "best". For best-checkpoint selection, run
`validate.py` over a few late checkpoints and pick the winner.

## Resume

Two flags, both optional, both work together:

```bash
# In sbatch/train_cwdm.sh — set these and re-submit:
RESUME_CHECKPOINT=/gpfs/.../wandb_logs/wandb/run-*-abc12345/files/checkpoints/synthrad_000200000.pt
RESUME_ID=abc12345         # 8-char wandb run id (from the URL or run dir)
```

- `RESUME_CHECKPOINT` → `--resume_checkpoint=...` — model weights, optimizer state
  (auto-loaded from sibling `opt000200000.pt`), and step counter (parsed from filename).
  Future checkpoints continue numbering from there.
- `RESUME_ID` → `--wandb_resume_id=...` — wandb continues the existing run instead of starting fresh.

You typically want both set for a clean resume.

## Manual run (without SLURM)

```bash
cd /home/minsukc/MRI2CT/baselines/cwdm
export WANDB_DIR=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs
bash run.sh
```

`run.sh` defaults to `MODE=train DATASET=synthrad CONTR=ct`. Edit the variables at the top of the file to override.

## What's logged to WandB

Per training step:
- `loss/MSE` (averaged wavelet MSE across the 8 subbands)
- `loss/mse_wav_lll`, `..._llh`, ..., `..._hhh` (each subband individually)
- `time/load`, `time/forward`, `time/total` (wall-clock per step)
- `info/lr`, `info/global_step`, `info/samples_seen`, `info/cumulative_time`
- `info/train_pct` (when `--lr_anneal_steps>0`, which the SLURM script sets)

Every 100 steps:
- `monitoring/ram_main_rss_gb`, `monitoring/ram_percent`, `monitoring/vram_alloc_gb`, `monitoring/vram_peak_gb`

Every `--val_interval` steps (default 20000):
- `val/mae_hu`, `val/psnr`, `val/ssim`, `val/grad_diff` (on the fixed val subject)
- `val/sample_image` (mid-slice grid)
- `val/subj_id`, `val/ddim_steps` (metadata)

Tags: `cwdm` (always) + any extra tags from `TAGS` in the SLURM script.

## Comparing against amix in the WandB UI

Project: `mri2ct`. Filter by tag:
- `tag:cwdm` for cWDM runs
- `tag:amix` for amix runs (or whatever amix uses — check)

Key names match (`val/mae_hu`, `val/psnr`, `val/ssim`, `info/lr`, `info/samples_seen`, etc.)
so single-chart side-by-side comparisons work out of the box.
