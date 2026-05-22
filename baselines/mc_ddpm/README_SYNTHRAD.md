# MC-IDDPM on SynthRAD — runbook

Adapter scripts that take the official author repo at `baselines/mc_ddpm/`
(Pan et al., *Med Phys 2024*, "Synthetic CT generation from MRI using 3D
Transformer-based Denoising Diffusion Model") and wire it into the MRI2CT
project's data layout, SLURM cluster, and WandB conventions, **without
editing any upstream file**.

The original repo's notebook (`MC-IDDPM main.ipynb`) is preserved as the
paper reference. We only add new files in this directory:

```
baselines/mc_ddpm/
├── data.py                  — paper-faithful MONAI compose for SynthRAD
├── trainer.py               — Trainer(BaseTrainer) wrapping the notebook's loop
├── train.py                 — argparse entry, MCDDPM_CONFIG, calls Trainer.train()
├── scripts/validate.py      — post-hoc full-volume sliding-window eval
├── README_SYNTHRAD.md       — this file
└── __init__.py              — sys.path shim so cloned `diffusion.*` imports work
```

Original upstream files (`diffusion/`, `network/`, `MC-IDDPM main.ipynb`,
`LICENSE`, etc.) are untouched.

---

## Prereqs

- `mrct` conda/micromamba env (Python 3.10 + PyTorch 2.x + MONAI 1.4). The
  cloned repo's `environment.yml` (Py 3.8 + PyTorch 1.9 + MONAI 0.7) is **not**
  used — we run against the project's modern stack.
- A40 GPU (≥ 16 GB). Worst-case VRAM peak at paper config (bs=4, pv=2) is
  ~13 GB, so the model has room to scale up if needed.
- WandB account if logging is enabled.

---

## Data layout (read-only, gpfs)

```
/gpfs/.../SynthRAD/1.5mm_registered_flat_masked/<subj_id>/
  ct.nii(.gz)
  moved_mr.nii(.gz)
  mask.nii(.gz)        (body mask)
  ct_seg.nii(.gz)      (optional)
```

Split file: `splits/center_wise_split.txt` — one `SPLIT_NAME SUBJECT_ID`
per line. Defaults: 427 train / 207 val.

---

## Submit a fresh run

```bash
sbatch sbatch/train_mcddpm.sh
```

⚠️ Use `sbatch`, **not** `bash sbatch/train_mcddpm.sh` — the self-submission
stanza checks `$SLURM_JOB_ID`; running via `bash` inside an interactive SLURM
allocation will skip the self-submit branch and execute training inline on
the interactive node ([[feedback_sbatch_from_interactive]]).

### Editable variables (top of `sbatch/train_mcddpm.sh`)

| var | default | meaning |
|---|---|---|
| `SPLIT_FILE` | `splits/center_wise_split.txt` | which subjects to train/val on |
| `LR` | `3e-5` | paper text page 6 (notebook uses `2e-5`; we follow paper text) |
| `WEIGHT_DECAY` | `1e-5` | paper text page 6 (notebook uses `1e-4`; we follow paper text) |
| `BATCH_SIZE` | `4` | volumes per batch; effective batch = `BATCH_SIZE * PATCHES_PER_VOL` |
| `PATCHES_PER_VOL` | `2` | random patches per volume (paper notebook value) |
| `LR_ANNEAL_STEPS` | `0` | 0 → constant LR (paper); >0 → cosine over N steps |
| `USE_AMP` | `False` | turn on after smoke-testing stability |
| `USE_CHECKPOINT` | `False` | gradient checkpointing in SwinVITModel |
| `DIFFUSION_STEPS` | `1000` | paper |
| `VAL_STEPS` | `25` | in-training viz sampler steps (cheap; full eval uses 50 via `validate.py`) |
| `EPOCHS` | `500` | paper |
| `STEPS_PER_EPOCH` | `500` | effective: 500 × 500 = 250k train iters ≈ 46 h on A40 |
| `SAVE_INTERVAL` | `5` | epochs — milestone cadence |
| `VAL_INTERVAL` | `5` | epochs — viz hook cadence |
| `VAL_SUBJ_ID` | `1THB011` | which subject to visualize each val |
| `RESUME_ID` | `""` | set to a wandb run ID (8 chars) to chain-resume |
| `TAGS` | `""` | comma-separated extra wandb tags |

---

## Resume after a SLURM cut

A 48-hour SLURM job at default config covers ~250k iters — basically the
full 500-epoch paper recipe in one job. If your job hits the wall-clock
or crashes earlier, chain-resume:

1. Open `sbatch/train_mcddpm.sh`.
2. Find your wandb run id (printed in the SLURM log; also in
   `$WANDB_DIR/runs/<timestamp>_<id>/`).
3. Set `RESUME_ID="<id>"`.
4. `sbatch sbatch/train_mcddpm.sh`.

The trainer:
- Searches `$WANDB_DIR/runs/*_<id>/` (and legacy `wandb/run-*-<id>/files/`)
  for `checkpoint_last.pt`.
- Restores model, optimizer, scheduler, `global_step`, `samples_seen`, and
  `elapsed_time`.
- Resumes from epoch `<last_saved_epoch> + 1`.
- Writes new milestones with epoch numbers higher than the resumed point.

You'll see this in the log:
```
[RESUME] 🕵️ Searching for Run ID: <id>
[RESUME] 🔄 Loading checkpoint: .../checkpoint_last.pt
[RESUME] ✅ Resumed from Epoch <N>
```

---

## Evaluate a checkpoint

```bash
# Paper-faithful eval (50 reverse steps, single MC run):
python baselines/mc_ddpm/scripts/validate.py \
    --checkpoint $WANDB_DIR/runs/<ts>_<id>/mcddpm_epoch00499.pt

# Paper-faithful with Monte-Carlo averaging (paper: N=5):
python baselines/mc_ddpm/scripts/validate.py \
    --checkpoint <ckpt> --mc_runs 5

# Smoke / debug (much cheaper):
python baselines/mc_ddpm/scripts/validate.py \
    --checkpoint <ckpt> --max_subjects 2 \
    --ddim_steps 10 --overlap 0.25 --sw_batch_size 4
```

Outputs land in `<ckpt_dir>/validate_<timestamp>/`:
- `<subj>/sample.nii.gz` and `target.nii.gz` (HU units, GT affine preserved).
- `validate_metrics.json` with per-subject + summary.
- A wandb run tagged `mc-iddpm,validate` with `validate_paper/*` and
  `validate_amix/*` aggregates.

### Two metric families

Per the project convention that distinct scales must have distinct keys
([[feedback_metric_naming]]):

| key family | clip range | span | use |
|---|---|---|---|
| `validate_paper/*` | `[-1024, 1650]` (paper) | 2674 | apples-to-apples with the paper's tables |
| `validate_amix/*`  | `[-1024, 1024]` (project) | 2048 | apples-to-apples with `src/amix/`'s `val/mae_hu` |

The amix family **re-clips** predictions back to `[-1024, 1024]` HU before
scaling to `[0, 1]`, so any predicted intensity above 1024 HU (paper
preserved up to 1650 HU) is clamped. This is the right comparison for
benchmarking against `src/amix/`.

---

## Where things land on disk

```
$WANDB_DIR/runs/<timestamp>_<run_id>/
├── checkpoint_last.pt         — rolling, overwritten each save
├── mcddpm_epoch<NNNNN>.pt     — preserved milestones
├── model_summary.txt          — param count + per-module breakdown
├── sessions/<timestamp_run>/  — per-resume session config + slurm.log link
└── validate_<ts>/             — outputs of scripts/validate.py
    ├── validate_metrics.json
    └── <subj_id>/
        ├── sample.nii.gz
        └── target.nii.gz
```

`$WANDB_DIR =
/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs`.

---

## WandB keys

Per train step (every `cfg.log_every`, default 50):
- `loss/total`, `loss/mse`, `loss/vb`
- `time/load`, `time/total`
- `info/lr`, `info/global_step`, `info/samples_seen`, `info/epoch`,
  `info/cumulative_time`, `info/train_pct` (if `lr_anneal_steps>0`)

Per epoch:
- `train/loss_total`, `train/loss_mse`, `train/loss_vb`
- `info/epoch_duration`, `info/val_duration`, `info/cumulative_time`,
  `info/global_step`, `info/epoch`, `info/samples_seen`

Per val (visualization-only):
- `val/sample_image` (4-panel: MRI | GT CT | Pred CT | |diff|)
- `val/subj_id`, `val/sampling_steps`

WandB summary at startup:
- `total_params`, `trainable_params`

Tags: `["mc-iddpm"]` + user-supplied. Validate adds `["validate"]`.

---

## Expected compute

At paper config (bs=4, pv=2, 500 epochs × 500 steps):
- ~666 ms/iter on A40 → ~46.3 hours wall-clock end-to-end.
- VRAM peak ~12.9 GB (44 GB A40 — comfortable).
- Single 48-h SLURM job covers the full paper recipe with margin.

Validation (paper-faithful, ddim=50, mc_runs=1, overlap=0.5):
- ~10-15 min/subject (rough; depends on padded volume size).
- For 207 val subjects: ~35-50 h total.
- With `mc_runs=5`: ~5× that — typically run as a separate batch.

---

## Known caveats / deliberate deviations from amix

| amix default | this baseline | why |
|---|---|---|
| CT clip `[-1024, 1024]` → `[0, 1]` | **CT clip `[-1024, 1650]` → `[-1, 1]`** | paper recipe; `clip_denoised=True` in `GaussianDiffusion` requires `[-1, 1]` |
| MRI minmax → `[0, 1]` | **MRI minmax → `[-1, 1]`** | same reason |
| `get_cached_transforms` + `get_random_crop` + `get_gpu_transforms` (full aug stack) | **Inline paper compose, NO augmentation, uniform random crop** | paper has no aug; project utilities are `[0,1]`-hardcoded |
| Full-volume `bs=1` training | **Patch-based `bs=4`, `pv=2` `(128,128,4)` crops** | paper is explicitly patch-based |
| In-training metric val hook (`val/mae_hu`, etc.) | **Visualization-only val** | metrics computed post-hoc in `scripts/validate.py` |
| `hu_range=2048` for metrics | **Two key families: `*_paper` (2674), `*_amix` (2048)** | preserves paper-faithful numbers while still allowing apples-to-apples vs amix |

### Structural caveat: 4 axial slices vs thorax

The paper used `(128, 128, 4)` for prostate (32-slice volumes), where a
4-slice patch covers ~12% of z. On our 1.5mm thorax data (up to ~280 slices
padded), 4 slices is ~1-2% of z. This is a structural disadvantage from
transferring the published recipe — long-axis structure (lung lobes,
vertebrae, mediastinum) gets minimal z-context per patch.

The paper justified the thin slab as an inference-cost optimization
(Table D3: 4-slice patches converge with 50 reverse steps; thicker
patches need 2-4× more). We kept it because we're answering the question
"does the published recipe transfer?", not "what's the best MC-IDDPM
on thorax?".

### Upstream quirk: `sample_kernel` 1-tuple

`SwinVITModel.__init__` does `self.sample_kernel = sample_kernel[0]` at
line 359, so the `sample_kernel` kwarg must be wrapped in an outer 1-tuple
— exactly what the notebook achieves with the trailing-comma assignment:
```python
sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),
```
We pass `sample_kernel=(([2,2,2],[2,2,1],[2,2,1],[2,2,1]),)` explicitly
in `trainer.py` and `scripts/validate.py` for clarity.

---

## File map

**New (in `baselines/mc_ddpm/`)**:
- `data.py`, `trainer.py`, `train.py`, `scripts/validate.py`,
  `README_SYNTHRAD.md`, `__init__.py`, `diffusion/__init__.py`,
  `network/__init__.py`, `scripts/__init__.py`
- `baselines/__init__.py`
- `sbatch/train_mcddpm.sh`

**Untouched upstream**: everything under `baselines/mc_ddpm/diffusion/`,
`baselines/mc_ddpm/network/`, `MC-IDDPM main.ipynb`, `LICENSE`, `README.md`,
`environment.yml`, `MRI_to_CT_brain_for_dosimetric/`.

---

## Cheat sheet

```bash
# Submit fresh
sbatch sbatch/train_mcddpm.sh

# Resume (edit RESUME_ID in the script first)
sbatch sbatch/train_mcddpm.sh

# Smoke a single iter (no wandb, no val)
WANDB_MODE=disabled python baselines/mc_ddpm/train.py \
    --wandb=False --epochs=1 --steps_per_epoch=1 \
    --val_interval=999 --save_interval=999 \
    --batch_size=1 --patches_per_volume=1 --num_workers=0

# Evaluate paper-faithful
python baselines/mc_ddpm/scripts/validate.py \
    --checkpoint <path> --mc_runs 5

# Evaluate quickly (smoke)
python baselines/mc_ddpm/scripts/validate.py \
    --checkpoint <path> --max_subjects 2 --ddim_steps 10 \
    --overlap 0.25 --sw_batch_size 4
```
