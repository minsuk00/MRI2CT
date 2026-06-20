# Cross-model bone comparison: is the undershoot universal?

**Source HTML:** _html/07_crossmodel_bone.html
**Date:** 2026-06-19 (full_eval_20260617 / unet_failure_20260619)
**TL;DR:** Bone undershoot is universal: all 6 models (regression and diffusion, sigmoid-capped and uncapped, including koalAI) predict bone too low, predict cortical bone much worse, and regress to the mean for dense bone. Uncapped models reach higher peak HU but still fall far short of cortical HU, confirming an information + objective limit, not an architecture limit. In every model, fixing air/soft gains more PSNR than fixing bone.

Same bone diagnosis as reports 05/06, run on all 6 models over the 207 center-wise validation subjects. Sign convention: error = pred - truth; negative bias = undershoot. Parity caveat: cwdm and maisi are far below training-sample budget (0.24x / 0.22x) and still training; read their numbers as lower bounds.

## Correctness (PASS)

For every model the oracle baseline reproduces the released body_psnr and body_mae_hu to <0.01.

| model | Δ base PSNR | Δ base bodyMAE |
| --- | --- | --- |
| unet | 0.0000 | 0.0000 |
| amix | 0.0000 | 0.0000 |
| koalAI | 0.0000 | 0.0001 |
| mcddpm | 0.0000 | 0.0001 |
| cwdm | 0.0000 | 0.0001 |
| maisi | 0.0000 | 0.0001 |

## 1. Every model undershoots bone

| model | parity | bone bias (HU) | cortical bias (HU) | bone MAE (HU) | % subj undershoot | % bone vox under | pred bone p99 | PSNR gain: air | PSNR gain: bone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet | 1.00x | -179.6 | -520.7 | 240.8 | 100.0 | 80.7 | 809.2 | 2.8 | 1.3 |
| amix | 1.00x | -182.1 | -528.8 | 239.7 | 100.0 | 80.8 | 786.8 | 3.0 | 1.2 |
| koalAI | ~0.78x | -151.3 | -489.0 | 233.0 | 98.6 | 71.1 | 1101.1 | 3.3 | 1.0 |
| mcddpm | 1.25x | -210.4 | -581.7 | 271.7 | 99.5 | 80.0 | 1011.1 | 2.2 | 1.2 |
| cwdm | 0.24x (under-parity) | -273.7 | -680.2 | 322.1 | 100.0 | 86.9 | 823.1 | 1.8 | 1.4 |
| maisi | 0.22x (under-parity) | -272.8 | -628.3 | 334.9 | 100.0 | 83.9 | 897.4 | 2.1 | 1.4 |

## 2. All models regress to the mean for dense bone

Mean predicted HU at each true-bone-HU level: every model flattens out far below the identity line as bone gets denser. koalAI/mcddpm climb a little higher but still collapse. The final point (true HU 2000-2900) is very sparse (~0.1% of bone voxels) so it is noisy.

## 3. No model reaches dense cortical HU

Sigmoid-capped regression models (U-Net, amix) top out ~880-1024 HU. Unbounded models (koalAI, mcddpm, cwdm, maisi) output higher and reach higher p99, but still land far below real cortical bone (GT bone mean 564, cortical >1024 up to ~2900). The cap is not the only limit; the information is.

## 4. In every model, air/soft are bigger PSNR levers than bone

The oracle (perfect one tissue, recompute reported metric) gives the same ranking for all models: air and soft beat bone, because bone is ~5% of voxels in every case and the metric clips it.

## 5. Note on per-label coverage (math)

The 35 CADS labels cover only ~82% of body voxels; the remaining ~18% is seg==0 inside the body, almost entirely unlabeled internal air (sinus / bowel gas / cavities CADS does not assign). External background masked to -1024 and excluded. Consequence: voxel-averaging the per-label MAEs reconstructs the MAE over labeled voxels, not the full body MAE. The air/soft/bone HU split tiles 100% of the body and does reconstruct the body MAE exactly (verified by the mass-conservation gate in report 05).

## 6. Conclusion

- Bone undershoot is universal. Every model (regression and diffusion, capped and uncapped, including koalAI) undershoots bone and cortical bone, and all regress to the mean for dense bone.
- Uncapped helps a little, not enough. koalAI/mcddpm reach higher peak HU than the sigmoid-capped U-Net/amix but still fall far short of cortical HU, consistent with an information limit.
- Same metric story everywhere. Air/soft are the bigger PSNR levers in every model; bone leads only on per-voxel / clinical accuracy.
- cwdm/maisi numbers are lower bounds (still under-parity) but show the same qualitative pattern.

## Reproducibility

Scripts: `src/evaluate/unet_failure/multimodel_extract.py` (per-model per-subject) and `multimodel_report.py` (this report). Data: `evaluation_results/unet_failure_20260619/multimodel_bone.csv` + `multimodel_calib.npz`. Oracle replicates `compute_metrics_body`; GT = raw `ct.nii` in canonical RAS.
