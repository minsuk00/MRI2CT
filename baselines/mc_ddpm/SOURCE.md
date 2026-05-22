# Source

Upstream: https://github.com/shaoyanpan/Synthetic-CT-generation-from-MRI-using-3D-transformer-based-denoising-diffusion-model
Vendored from commit: `5bbe985578796505a709c03e5e2126c1096a1e57` (branch `main`)
Date vendored: 2026-05-21

## Local additions (not in upstream)

Project glue for the MC-IDDPM baseline in MRI2CT — none of the paper-defined model / diffusion code was modified.

- `__init__.py`, `diffusion/__init__.py`, `network/__init__.py`, `scripts/__init__.py` — package stubs so `baselines.mc_ddpm.*` imports cleanly from the MRI2CT side.
- `data.py` — MONAI-cached MRI/CT pipeline + `PATCH = (128, 128, 4)` definition shared by `trainer.py` and `scripts/validate.py`.
- `train.py` — CLI entry point (argparse → `EXPERIMENT_CONFIG` overrides → `Trainer.train()`). Mirrors the layout of MRI2CT's other trainers.
- `trainer.py` — `Trainer(BaseTrainer)`. Training step is byte-equivalent to the upstream notebook's `train()` body; the wrapper adds WandB logging, two-tier checkpointing via `BaseTrainer.save_checkpoint`, and a viz-only `validate()` hook (single-patch sample + mid-slice grid; full metrics live in `scripts/validate.py`).
- `scripts/validate.py` — post-training full-volume sliding-window sampling via MONAI `sliding_window_inference`. Computes MAE/PSNR/SSIM/Bone Dice per subject and writes NIfTI predictions.
- `README_SYNTHRAD.md` — notes on the SynthRAD data layout.

## License

See `LICENSE` (MIT, © 2023 shaoyanpan). Local additions inherit the MIT terms.
