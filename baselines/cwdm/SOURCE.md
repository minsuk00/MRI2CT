# Source

Upstream: https://github.com/pfriedri/cwdm
Vendored from commit: `1e4a1e57eec3a5d0f1375c4cde4d38b837c0fce0` (branch `main`)
Date vendored: 2026-05-21

## Local modifications to upstream files

- `DWT_IDWT/DWT_IDWT_layer.py` — minor changes for 3D-DWT compatibility with our data shapes.
- `guided_diffusion/gaussian_diffusion.py` — adjustments to support our training/inference path.
- `guided_diffusion/train_util.py` — added WandB logging mirror of TensorBoard scalars, added `_validate_one` (single-subject DDIM val sample + metrics + mid-slice grid) fired every `val_interval` steps, and a free `_log_train_sample` (mid-slice viz of the in-loop `sample_idwt` on the current training batch) for overfitting comparison.
- `scripts/train.py` — added `--wandb_resume_id`, `--val_subj_id`, `--val_interval`, `--val_ddim_steps`, and the auto-discovery path that looks under `wandb/run-*-<resume_id>/files/checkpoints/synthrad_last.pt` so SLURM chain-resumes work without a hard-coded path.
- `scripts/sample.py` — minor adjustments for our data layout.
- `run.sh` — adjusted for our paths / environment.

## Local additions (not in upstream)

- `guided_diffusion/synthradloader.py` — SynthRAD MR↔CT dataset loader (cached MONAI pipeline shared with MRI2CT).
- `scripts/validate.py` — post-training full-volume eval.
- `README_SYNTHRAD.md` — notes on the SynthRAD data layout.

## License

See `LICENSE` (MIT, © 2024 Paul Friedrich). Local modifications inherit the MIT terms.
