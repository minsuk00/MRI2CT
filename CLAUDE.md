# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MRI2CT synthesizes CT images from MRI scans using deep learning. The main approach uses a U-Net translator with a frozen pre-trained **Anatomix** feature extractor for perceptual/segmentation losses. Baselines include a standard U-Net, a MAISI diffusion ControlNet, a BabyUNet segmentation teacher, and the **KoalAI** SynthRAD2025 winner (`baselines/koalAI/`, nnsyn fork — see `baselines/koalAI/TODO.md`).

## Environment

```bash
micromamba activate mrct      # main repo + amix/unet/maisi/baby_unet + TotalSegmentator
micromamba activate koalai    # nnsyn fork (only for KoalAI baseline: convert/preprocess/train/predict)
micromamba activate pyradplan # pyRadPlan dose-eval downstream task ONLY (pyradplan/ dir)
```

The two main envs are kept separate because both register a package named `nnunetv2`: `mrct` has upstream 2.5.2 (required by TotalSegmentator), `koalai` has the nnsyn fork. Don't pip-install across them.

`pyradplan` is a third, isolated env for the pyRadPlan dose-evaluation downstream task (`pyradplan/`). It MUST stay separate: pyRadPlan requires `numpy>=2` (and drags in `numba`/`numpydantic` that force-upgrade numpy), which breaks `mrct`'s numpy-1.26 world (scikit-image / tptbox). Never `pip install pyRadPlan` into `mrct`. See `pyradplan/README.md`.

Data lives on GPFS (read-only): `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked/`

Linting: `ruff check src/` (E402 is globally ignored via `pyproject.toml`).

## Running Training

```bash
# Anatomix translator (main model)
python src/amix/train.py

# With CLI overrides
python src/amix/train.py --dice_w 0.1 --epochs 1000 --split_file splits/thorax_center_wise_split.txt

# Resume from WandB checkpoint
python src/amix/train.py --resume_id <wandb_run_id>

# UNet baseline
python src/unet_baseline/train.py

# MAISI diffusion baseline
python src/maisi_baseline/train.py

# BabyUNet teacher (12-class CT organ segmentation)
python src/seg_baby_unet/train.py

# Via SLURM (A40 GPU)
sbatch sbatch/train_amix_v1_4.sh   # also: train_unet.sh, train_maisi.sh
```

### KoalAI baseline (separate `koalai` env, per-region models, dual-fold)

Two splits live in one preprocessed dataset as separate folds — fold 0 = center-wise (OOD test), fold 1 = random (i.i.d.). Trained seg model is the perceptual extractor for synth's MAP loss; synth needs seg of the same fold to be done first. Full workflow in `baselines/koalAI/TODO.md`.

```bash
# Pilot: thorax fold 0 (centerwise) seg → synth → val inference
REGION=thorax                       sbatch sbatch/koalai_train_seg.sh
REGION=thorax                       sbatch sbatch/koalai_train_synth.sh   # after seg done
REGION=thorax SUBSET=val            sbatch sbatch/koalai_predict.sh       # SUBSET=val|test, default test

# Fold 1 (random) for any step: add FOLD=1
REGION=thorax FOLD=1                sbatch sbatch/koalai_train_seg.sh
```

## Running Evaluation

```bash
# Multi-model metric eval (MAE/PSNR/SSIM/Bone Dice) over a split's val set
python src/evaluate/evaluate_models.py \
    --split_file splits/thorax_center_wise_split.txt \
    --checkpoints amix:/path/to/amix.pt unet:/path/to/unet.pt maisi:/path/to/maisi.pt

# Side-by-side qualitative figures + NIfTI export (1 subject per region)
python src/evaluate/visualize_predictions.py \
    --checkpoints amix:/path/to/amix.pt unet:/path/to/unet.pt

# Organ segmentation eval via TotalSegmentator (edit PRED_DIR inline)
python src/evaluate/total_seg.py

# PCA feature visualization
python src/evaluate/compare_amix_features.py
```

## Architecture

### Source Structure

- `src/common/` — shared modules used by all trainers:
  - `config.py`: `DEFAULT_CONFIG` dict with all hyperparameters; `Config` (SimpleNamespace wrapper)
  - `data.py`: subject-discovery helpers (`get_subject_paths`, `get_split_subjects`, `get_region_key`) + MONAI 3-stage pipeline (`get_cached_transforms`, `get_random_crop`, `get_gpu_transforms`, `gpu_augment_batch`, `default_monai_cache_dir`, `build_data_dicts`). Used by all trainers and eval scripts. The legacy torchio version is preserved at `src/_archive/data_torchio.py` for reference.
  - `loss.py`: `CompositeLoss` (L1 + SSIM + optional Anatomix perceptual + optional teacher Dice + optional bone Dice). **When the perceptual loss is on (`perceptual_w > 0`), set `ssim_w = 0`** — perceptual replaces SSIM as the structural-similarity term on top of L1; using both double-counts structure. Dice loss is a per-class weighted macro-mean over **all** classes incl. background (bone class uses `dice_bone_w`, others `dice_w`).
  - `trainer_base.py`: `BaseTrainer` (seeding, WandB init, checkpoint resume, `_log_monitoring()` for RAM/VRAM/timings)
  - `utils.py`: `anatomix_normalize()`, `compute_metrics()`, `cleanup_gpu()`, `send_notification()`, `get_ram_info()` (parent + recursive children PSS via psutil)
- `src/amix/` — anatomix translator trainer
- `src/unet_baseline/` — plain U-Net baseline
- `src/maisi_baseline/` — MAISI diffusion ControlNet baseline
- `src/seg_baby_unet/` — teacher segmentation network
- `src/evaluate/` — NIfTI export and metric evaluation scripts
- `src/preprocess/` — MRI-CT registration, resampling, segmentation pipelines
- `anatomix/` — local package with pre-trained U-Net feature extractor (versions v1, v1_2, v1_3)
- `splits/` — train/val split files (`SPLIT_NAME SUBJECT_ID` per line); `splits/koalai/` holds per-region derivations of both center_wise and random splits
- `baselines/koalAI/` — cloned nnsyn fork + our `mri2ct_scripts/` integration helpers (only the helpers + wandb hooks are our code; rest is upstream)
- `baselines/koalAI-seg/` — git worktree of `koalAI` on the `nnunetv2` branch (used by seg training; same wandb-hook patch applied)

### Training Data Flow

MONAI 3-stage pipeline (`src/common/data.py`):

1. **Cached CPU preprocessing** (`get_cached_transforms` → `monai.data.PersistentDataset`):
   load NIfTIs → enforce RAS → CT clip-and-scale (e.g. -1024..1024 → 0..1) → MRI minmax or percentile (0–99.5) → snapshot pre-pad shape → pad-end to ≥patch_size and to multiple of `res_mult` → uint8 mask hygiene. Cached at `/tmp/mri2ct_<USER>_monai_cache` (single shared dir; PersistentDataset hashes data+transform spec, so different trainers get separate entries automatically).
2. **CPU random crop** (`get_random_crop`, wrapped via `monai.data.Dataset(base, transform=crop)`):
   `RandWeightedCropd` (body-mask-weighted) or `RandSpatialCropSamplesd`. Yields `num_samples=patches_per_volume` uniform 128³ patches per volume in DataLoader workers. Skipped entirely by MAISI (full-volume training).
3. **GPU augmentation** (`get_gpu_transforms` + `gpu_augment_batch`):
   per-item, on device — RandAffined → RandFlipd×3 → RandBiasFieldd (MRI) → RandAdjustContrastd (MRI) → RandGaussianNoised (MRI) → ScaleIntensityd. Operates on either 128³ patches (amix/unet) or full padded volumes (MAISI).

DataLoader uses default `list_data_collate` (uniform-shape patches stack cleanly; for MAISI `batch_size=1` so no collation issue). Effective batch size for amix/unet = `batch_size × patches_per_volume`.

Validation: `MONAI sliding_window_inference` walks `val_patch_size`³ patches over the full padded volume → unpad to `original_shape` → MAE, PSNR, SSIM, Bone Dice logged to WandB. MAISI also runs the VAE encode/decode via sliding window (`_encode_sliding_window`, `_decode_sliding_window`).

### Configuration Pattern

All trainers define an `EXPERIMENT_CONFIG` list at the top of `train.py` — each dict overrides `DEFAULT_CONFIG`. Multiple experiments run sequentially in one job. CLI args further override the first experiment's config.

## Development Standards (from GEMINI.md)

- **Minimalism**: Do not refactor working code. Minimal, simple changes preferred.
- **Propose before editing**: Suggest changes in chat before modifying files.
- **Verify paths**: Many paths are hardcoded to `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/`. Verify before running.
- **Log research changes**: Record hyperparameter shifts in `research_notes.md` (Date | Action | Reason).
- **Git**: Work on branches; commit after successful edits.
