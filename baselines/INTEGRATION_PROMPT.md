# Integration Prompt — New Generative/Diffusion Baseline → MRI2CT

This file is a **prompt** to paste into a fresh Claude conversation after cloning a new
generative MRI→CT baseline paper repo into `baselines/<NAME>/`. It tells the agent
how to adapt the cloned code so it (a) trains on SynthRAD with the project's
normalization conventions, (b) logs to WandB with the project's key conventions,
(c) resumes cleanly across SLURM 48-hour cuts, and (d) can be evaluated against
amix using the same metric definitions.

It is **NOT** a runbook for the user to follow manually — it is a prompt for *Claude*
to perform the adaptation.

---

## How to use

1. Clone the new baseline into `/home/minsukc/MRI2CT/baselines/<NAME>/`.
2. Start a fresh Claude Code session in `/home/minsukc/MRI2CT/`.
3. Paste everything between the `---PROMPT START---` and `---PROMPT END---` markers,
   replacing `<NAME>` with the directory name and `<PAPER_TITLE>` with the paper.
4. Add one sentence describing what the baseline conditions on (MR only? MR + seg?)
   if it isn't immediately obvious from the paper.
5. Claude will read the code, ask 2–3 clarifying questions, propose a plan, and after
   approval execute the adaptation + smoke tests.

---

```
---PROMPT START---
You are integrating a new generative MRI→CT baseline at `baselines/<NAME>/` (paper:
<PAPER_TITLE>) into the MRI2CT project. The reference baseline that this project's
runs are compared against is `src/amix/`; your integration must produce a baseline
that sits alongside amix runs in the same WandB project with comparable logging,
identical normalization, and the same resume ergonomics.

Project shared utilities you MUST reuse (do not reinvent or fork these):
  - `src/common/data.py`
        get_split_subjects(split_file, split_name)
        build_data_dicts(root_dir, subjects, load_seg=False, load_body_mask=True)
        get_cached_transforms(...)   # MONAI cached pipeline; returns Compose
        default_monai_cache_dir()
  - `src/common/utils.py`
        compute_metrics(pred, target, hu_range=2048)
        compute_metrics_body(pred, target, mask, hu_range=2048)
        unpad(data, original_shape, offset=0)
  - `src/common/config.py`         # canonical defaults if you need to read them

Cluster / project conventions you MUST follow:
  - WandB project: `mri2ct`. Tag every run with `<NAME>` (lowercase, hyphenated).
  - WandB root directory on disk: `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs`
    (export `WANDB_DIR` to this in the SLURM script).
  - SLURM partition `spgpu`, single A40 GPU, max 48-hour wall clock per job.
  - Checkpoints land under `$WANDB_DIR/wandb/run-<ts>-<id>/files/checkpoints/`.
  - Data root: `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked/`.
    Each subject has `ct.nii`, `moved_mr.nii`, `mask.nii`, optionally `ct_seg.nii`.
  - Default split file: `/home/minsukc/MRI2CT/splits/center_wise_split.txt`.
  - Original paper paths in the cloned repo must still work via a `--dataset=<original>`
    branch. All your additions are guarded by `--dataset=synthrad`.

────────────────────────────────────────
A. UNDERSTAND THE BASELINE FIRST
────────────────────────────────────────
Before editing anything, summarize in ≤300 words:
  - Model architecture (U-Net / transformer / latent-diffusion / GAN / etc.).
  - Conditioning interface (channel concat, cross-attention, ControlNet, classifier-
    free guidance, FiLM, …).
  - What latent / image / wavelet space the model operates in. If there's a
    pretrained autoencoder/feature extractor, note where its weights live.
  - The published training recipe: lr, batch size, total iterations, augmentation,
    expected input normalization, AMP/fp16.
  - The sampler at inference time (DDPM/DDIM/EDM/PNDM/etc.) and step count.
  - Spatial-shape constraints (divisibility, fixed bottleneck attention size,
    hardcoded literals anywhere).

Flag any constraints that conflict with our SynthRAD data:
  - Per-subject shapes vary; if the architecture requires a fixed input size, decide
    between (i) padding all subjects to a global max, (ii) center-cropping, or
    (iii) downsampling. Surface the trade-off; don't silently pick.
  - SynthRAD subjects range up to (393, 261, 262) at 1.5mm; padded to multiple of 32
    that's up to (416, 288, 288). The biggest subject's padded VRAM footprint is
    what gates whether the baseline fits on A40 (44 GB).
  - If the paper uses `attention_resolutions` or attention bottlenecks that lock the
    input size, surface this; usually you'll need to disable them.

────────────────────────────────────────
B. DATA INTEGRATION (parity with amix)
────────────────────────────────────────
Add `baselines/<NAME>/<loader>.py` (filename matching the baseline's convention) that
wraps `monai.data.PersistentDataset` using the project helpers above:

    subjects = get_split_subjects(split_file, split_name)
    dicts    = build_data_dicts(root_dir, subjects, load_body_mask=True)
    xform    = get_cached_transforms(
                  patch_size=128, res_mult=<as needed by model>,
                  enforce_ras=True,
                  ct_range=(-1024, 1024),     # HU range; clipped + scaled to [0,1]
                  mri_norm='minmax',           # per-volume minmax to [0,1]
                  load_seg=False, load_body_mask=True,
              )
    ds       = PersistentDataset(data=dicts, transform=xform, cache_dir=default_monai_cache_dir())

Per-item dict keys returned by the loader (or via collate in your trainer):
  'mri'             (1, D, H, W) float32, padded
  'ct'              (1, D, H, W) float32, padded
  'body_mask'       (1, D, H, W) uint8 / float32
  'subj_id'         str
  'original_shape'  long tensor[3]  (pre-pad spatial dims, for unpad at eval)
  'ct_affine'       (4, 4) float32  (for NIfTI export at eval)

Constraints:
  - Full padded volume per item, bs=1. No random crop unless the paper is explicitly
    patch-based (most diffusion papers are full-volume).
  - No augmentation by default — match the paper. If the paper uses augmentation,
    use it (and document it).
  - Normalization is NON-NEGOTIABLE: CT clipped to [-1024, 1024] HU → [0, 1]; MR
    per-volume minmax → [0, 1]. Deviating breaks fair comparison against amix.
  - `res_mult` value depends on the model's downsampling depth; e.g. 32 = 1 DWT
    halving + 4 U-Net downsamples, or 8 for a 3-level U-Net.

────────────────────────────────────────
C. PATCHING THE ORIGINAL CODEBASE (additive only)
────────────────────────────────────────
Every patch must be additive: add `if dataset == 'synthrad':` branches; leave the
original `if dataset == 'brats':` (or whatever) branch byte-equivalent. Default the
`dataset` flag to `'synthrad'` in the argparser since this fork targets SynthRAD;
default any modality/contrast flag accordingly (e.g. `contr='ct'`).

Files typically patched:
  - The trainer's data-iterator block: dispatch on `dataset` and use the new loader.
  - The trainer's forward/loss call: thread CT as target, MR as condition (for MR→CT).
  - The argparser: add new defaults (`split_file`, `patch_size`, `res_mult`,
    `ct_range_lo/hi`, `mri_norm`, wandb args, val-hook args).
  - The sampling script: dataset-aware noise shape (DERIVE FROM COND TENSOR, never
    hardcoded literals); dataset-aware subject-id extraction; optional body-mask
    post-multiply (mirrors the original's background cleanup, if any).
  - `run.sh` / similar: add synthrad branch with the right `IN_CHANNELS`, `DATA_DIR`,
    `SPLIT_FILE`.

Common pitfalls to fix upstream as you encounter them (don't paper over):
  - Hardcoded spatial dimensions in noise sampling or model construction. Make them
    dynamic.
  - Hardcoded total timestep counts in sample loops (e.g. `time=1000`). If the paper
    supports respaced/accelerated sampling, ensure the loop respects the respaced
    count.
  - Shape-dependent precomputed matrices (e.g. wavelet filters) that silently
    truncate when one dimension is larger than expected.

────────────────────────────────────────
D. WANDB LOGGING (mandatory keys for amix parity)
────────────────────────────────────────
WandB init:
  wandb.init(
      project='mri2ct',
      name=args.wandb_run_name or None,
      tags=['<NAME>'] + (args.wandb_extra_tags or '').split(','),
      config=vars(args),
      id=args.wandb_resume_id or None,
      resume='allow' if args.wandb_resume_id else None,
      dir=...,  # optional; WANDB_DIR env var should be set in sbatch
  )

Configure the in-repo logger AFTER wandb.init so it never writes to `./results/`.
Point it at `wandb_run.dir` (or the tensorboard logdir if TB is requested instead).

**Required keys per training step:**
  - `loss/<primary_loss>` — use the paper's name (e.g. `loss/MSE`, `loss/L1`, etc.).
    For multi-term losses, log each term as a separate key.
  - `time/load`, `time/forward`, `time/total`
  - `info/lr`, `info/global_step`, `info/samples_seen`, `info/cumulative_time`
  - `info/train_pct` if `lr_anneal_steps > 0`

**Required `wandb_run.summary` entries at startup:**
  - `model/total_params`, `model/trainable_params`, `model/state_dict_mb`

**Required val-hook keys (see section E):**
  - `val/mae_hu`, `val/psnr`, `val/ssim`, `val/grad_diff`
  - `val/sample_image` (matplotlib mid-slice grid: MRI / GT CT / Pred CT / Residual)
  - `val/subj_id` and a sampler-step-count key (e.g. `val/sampling_steps`)

Do NOT add `monitoring/ram_*` / `monitoring/vram_*` unless the user explicitly asks.

────────────────────────────────────────
E. IN-TRAINING VAL HOOK (cheap sanity check)
────────────────────────────────────────
Every `--val_interval` steps (default 20000): sample a single fixed val subject
(default `1THB011`) using a reduced-step sampler (default 50 steps via the paper's
accelerated-sampling mechanism — DDIM, EDM, PNDM, whatever the paper supports).
Compute MAE_HU / PSNR / SSIM / grad_diff via `compute_metrics(pred, target, hu_range=2048)`,
log to wandb.

Implementation details:
  - This is a divergence check, not model selection. Don't make it expensive.
  - `model.eval() / model.train()` around the call. Wrap in try/except so a val
    failure never kills training.
  - UNPAD the prediction back to `original_shape` BEFORE computing metrics; the
    padded zeros would otherwise dilute the score. Use `unpad()` from utils.py.
  - If a body mask is present, multiply prediction by mask before metrics. Use
    `compute_metrics_body` for the masked variant.
  - HU range for the MAE: 2048 (because we normalize CT to [-1024, 1024] → [0, 1]).

────────────────────────────────────────
F. FULL VALIDATION SCRIPT (paper-faithful)
────────────────────────────────────────
Add `baselines/<NAME>/scripts/validate.py` that runs the paper's sampler at FULL
step count (e.g. DDPM-1000 for a 1000-timestep diffusion model) over the entire
val split. Default `--ddim_steps` (or equivalent) should be the paper's eval value,
NOT the in-training reduced value.

For each subject:
  - Load via the same MONAI pipeline as training.
  - Run the paper's sampler.
  - Inverse-transform (IDWT / VAE-decode / whatever).
  - Unpad to `original_shape`.
  - Optionally post-mask with body mask.
  - Compute `compute_metrics(...)` and (if mask present) `compute_metrics_body(...)`.
  - Save `<subj>/sample.nii.gz` and `target.nii.gz` with `batch['ct_affine']`, NOT
    `np.eye(4)`, so the NIfTI lines up with the GT in any viewer.

Aggregate to `validate_metrics.json` (per-subject + summary). Wandb init with
`tags=['<NAME>', 'validate']` and log aggregates to `validate/*`.

────────────────────────────────────────
G. SLURM SUBMISSION SCRIPT
────────────────────────────────────────
Create `sbatch/train_<NAME>.sh`. Copy directives from an existing project sbatch
(e.g. `sbatch/train_amix_v1_4.sh`):
  - 1× A40, partition `spgpu`, account `jjparkcv98`, 48-hour wall clock.
  - Output to `/home/minsukc/MRI2CT/slurm_logs/<ts>_<job_name>_<jobid>.log`.
  - `export WANDB_DIR=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs`
  - `micromamba activate mrct` (or the paper's env if it can't run in mrct — flag
    the conflict to the user, don't silently switch envs).
  - Top-of-file editable variables: `SPLIT_FILE`, `LR`, `LR_ANNEAL_STEPS`,
    `SAVE_INTERVAL` (milestones), `SAVE_LAST_INTERVAL` (rolling last; see section H),
    `VAL_INTERVAL`, `VAL_SUBJ_ID`, sampler-step-count, `RESUME_ID`,
    `RESUME_CHECKPOINT`, `TAGS`.
  - Use ABSOLUTE PATHS for `SPLIT_FILE` and `DATA_DIR`; the script `cd`s into
    `baselines/<NAME>/` and relative paths break.
  - If the script's self-submission stanza expects `bash` invocation, document this
    in the runbook — `sbatch` direct invocation bypasses the derived job-name logic.

────────────────────────────────────────
H. CHECKPOINT CADENCE + AUTO-RESUME (two-tier, amix-style)
────────────────────────────────────────
SLURM jobs cap at 48 hours, so we need cheap, frequent state-saving. Mirror amix:
two save cadences, one for cheap crash-recovery, one for paper-style preserved
milestones.

  --save_interval=5000        Preserved milestones: <dataset>_<NNNNNN>.pt +
                              opt<NNNNNN>.pt. Written every save_interval steps,
                              never overwritten. Used for final eval and ablation.
  --save_last_interval=1000   Rolling latest: writes 3 files, OVERWRITTEN each fire:
                                <dataset>_last.pt   (model state_dict)
                                optlast.pt          (optimizer state)
                                last_step.txt       (just the integer step number)

So a typical 48-hour job ends with ~9 milestones (5000, 10000, …, 45000) +
ONE rolling last (e.g. at step ~47000 if save_last_interval=1000 and the
SLURM-kill came mid-interval). On crash/kill you lose at most one
save_last_interval's worth of work — typically ~1 hour.

Auto-resume entry-point logic: when `--wandb_resume_id=<id>` is set and
`--resume_checkpoint=""`, search BOTH dir patterns (online and offline):

  $WANDB_DIR/wandb/run-*-<id>/files/checkpoints/
  $WANDB_DIR/wandb/offline-run-*-<id>/files/checkpoints/

Discovery order:
  1. First check for `<dataset>_last.pt`. If found, use it AND set the resume step
     by reading the `last_step.txt` sidecar (NOT by parsing the filename, because
     "last" has no digits). For the optimizer, look for `optlast.pt`, not
     `opt<step>.pt`.
  2. Otherwise fall back to globbing `<dataset>_<digits>.pt`, sort lexicographically
     (zero-padded → numeric sort), pick the highest. Resume mechanism then uses the
     paper's existing filename-parsing path.

Implementation requires two small patches to the paper's existing resume code:
  - `_load_and_sync_parameters` (or equivalent): if `os.path.basename(ckpt).endswith('_last.pt')`,
    read step from sibling `last_step.txt` instead of parsing the filename.
  - `_load_optimizer_state` (or equivalent): if loading a `_last.pt` model, the opt
    file is `optlast.pt`, not `opt<NNNNNN>.pt`.

And add a `save_last()` method called inside the training loop on
`step % save_last_interval == 0`, writing the 3 files above. It MUST overwrite (no
unique step suffix) so disk doesn't balloon — only one `*_last.*` triple ever
exists per run.

Sbatch ergonomics: expose `RESUME_ID` (and optionally `RESUME_CHECKPOINT` for
explicit overrides) at the top of the SLURM script. On chained resubmits the user
edits one line:
  RESUME_ID="<8-char-wandb-id>"
and `sbatch` again. The script passes `--wandb_resume_id=$RESUME_ID` plus an empty
`--resume_checkpoint`, auto-discovery does the rest.

Off-by-one nit (pre-existing in many DDPM codebases): the paper's training loop
typically increments `self.step` at the END of each iteration. So a milestone
named `<dataset>_000005.pt` and a `last_step.txt` containing "4" may represent
the same trained state (after 4 completed iters, ready for the 5th). This is
upstream convention, not a bug. The resume math is self-consistent: load
milestone-5 → resume_step=5 → next iter is the 5th + 1 = 6th; load last.pt with
sidecar=4 → resume_step=4 → next iter is the 1st local + 4 = 5th. Same place
in the optimization trajectory; the filename interpretation just differs.

────────────────────────────────────────
I. END-TO-END SMOKE TEST ON THIS NODE
────────────────────────────────────────
Before declaring done, run these on the GPU at this node:

  1. Single-iter train on the smallest train subject, no wandb, no val,
     `lr_anneal_steps=1`. Forward + backward + save must complete cleanly.
     Tests for shape-handling bugs in the model.

  2. ~4-iter train with `WANDB_MODE=offline` + `WANDB_DIR=/tmp/<NAME>_wb`,
     `val_interval=2`, reduced-step val, `save_interval=4`. Verify:
       a. WandB offline dir contains expected keys (`info/`, `loss/`, `val/`,
          `time/`, `model/`).
       b. Val hook fires and logs non-NaN metrics.
       c. A checkpoint lands at step 4.

  3. validate.py smoke on 2 val subjects with a reduced step count for speed.
     Verify:
       a. `validate_metrics.json` exists with per-subject and summary blocks.
       b. NIfTI outputs have the SAME spatial shape as the original pre-pad input
          volumes (print and confirm).
       c. WandB logs aggregate to `validate/*`.

  4. Resume smoke: re-run train.py with `--wandb_resume_id=<id-from-step-2>` and
     empty `--resume_checkpoint`. Verify in the log:
       a. `[resume] auto-discovered checkpoint: <path>` is printed.
       b. The step counter resumes from the resumed file's step number.
       c. The next saved checkpoint is numbered higher than the resumed step.

  5. Worst-case-OOM test: scan all train subjects, compute padded voxel counts,
     pick the largest. Run 1 train iter on JUST that subject with gradient
     checkpointing (or whatever memory-saver the paper exposes) enabled. If it
     OOMs anyway, surface the numbers and ask the user before doing anything
     drastic (downsample, fp16, etc.).

  6. Per-iter timing on the largest subject: run 20 iters, divide wall time by 20
     minus setup. Report the number. Sanity-check vs the 48-hour SLURM limit.

Report which tests passed and which surfaced bugs. Don't claim "end-to-end works"
until all six pass.

────────────────────────────────────────
J. WRITE A USER-FACING RUNBOOK
────────────────────────────────────────
Finally, write `baselines/<NAME>/README_SYNTHRAD.md`. Sections:
  - Prereqs (env, GPU, dependencies).
  - Data layout (gpfs root, splits file format).
  - Submit fresh run (one-liner + variables-to-edit table).
  - Resume after SLURM cut (set RESUME_ID, resubmit).
  - Evaluate a checkpoint (validate.py invocation).
  - Where outputs land on disk.
  - WandB keys logged.
  - Expected wall-clock + total compute.
  - Known caveats / upstream bugs you fixed in-tree.
  - File map (new vs patched vs untouched).
  - Cheat sheet.

────────────────────────────────────────
K. INVARIANTS — DO NOT VIOLATE
────────────────────────────────────────
  - The original paper's data path must still work via `--dataset=<original>`.
  - Don't add features beyond what's listed here. No best-checkpoint tracking, no
    EMA shadows, no exotic augmentation. Paper-faithful is the goal.
  - Don't add `monitoring/ram_*` / `monitoring/vram_*` unless asked.
  - Don't deviate from amix's normalization (CT [-1024, 1024] HU, MR minmax).
  - Don't enable `--use_fp16=True` by default unless you've ACTUALLY wired
    `torch.cuda.amp.autocast` through the model forward — many papers' fp16
    plumbing is incomplete and the flag does nothing useful.
  - Don't add checkpoint cleanup / "keep last N" by default — gpfs has space.
  - Save checkpoints to gpfs (via wandb_run.dir under WANDB_DIR), NOT to the
    home directory or to `./results/`.

────────────────────────────────────────
L. WHEN UNSURE — ASK FIRST
────────────────────────────────────────
Use AskUserQuestion for ambiguous decisions:
  - Conditioning details (MR only? MR + seg? What channel ordering?).
  - Whether to deviate from the paper's recipe (resolution, augmentation, lr).
  - Whether to commit / push the changes (default NO).

Surface trade-offs explicitly. Don't silently downsample / crop / change shapes.
---PROMPT END---
```

---

## Notes for the user

- Hand the agent the cloned-repo directory plus a one-sentence description of the
  paper. The agent reads the paper PDF if it's in `baselines/<PaperName>.pdf`.
- Default behavior should be paper-faithful. Deviations only when forced by data
  shape, hardware, or comparison fairness (e.g. matching amix's normalization).
- The agent should produce a runbook in `baselines/<NAME>/README_SYNTHRAD.md` so
  you can launch + resume + eval without re-reading the integration prompt.
