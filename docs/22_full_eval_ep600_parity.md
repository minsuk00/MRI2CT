# MRI to CT Budget-Parity Evaluation (unet@ep600 vs koalAI)

**Source HTML:** _html/full_eval_ep600_parity.html
**Date:** 2026-06-04
**TL;DR:** At matched training budget (~2.4-2.5M samples), unet@ep600 beats koalAI on body-masked MAE (macro 32HU vs 39HU; micro 34HU vs 40HU) and on bone Dice across every region; the gap is significant (paired Wilcoxon p=4.2e-19) and not budget-limited (unet@ep600 ≈ unet@ep799 full 3.2M).

Center-wise validation split, 207 subjects, 5 regions, matched training budget (~2.4-2.5M samples). Both models scored at near-equal training budget: unet@ep600 (~2.4M samples, 0.75x) vs koalAI (~2.5M total, ~0.78x). This isolates method quality from training-budget difference.

## Models and checkpoints (training-sample parity)

| Model | Checkpoint | Samples | vs 3.2M | Note |
| --- | --- | --- | --- | --- |
| unet | 9xmodnhn/unet_baseline_epoch00600.pt (ep600) | ~2,400,000 | 0.75x | budget-matched to koalAI |
| koalAI | per-region fold_0 checkpoint_final.pth | ~500,000/region (~2.5M total) | ~0.78x | per-region; pre-generated |

## Track A: amix-clip [-1024, 1024] (hu_range 2048)

Body-masked metrics (first 3 columns) are the primary, fair comparison: voxels outside the body mask zeroed. Hard Bone Dice and Hard Dice (mean over all 11 foreground organ classes) via the Baby-UNet teacher, computed on argmax labels following dice-score-3d convention (both absent -> 1.0, one absent -> 0.0, else 2|A∩B|/(|A|+|B|)). Full-volume MAE/PSNR/SSIM columns are secondary and penalize models that flatten background: koalAI sets outside-body to constant -1000 HU, so its full-volume SSIM looks low (~0.5) though its body-masked SSIM (~0.9) matches. Mean ± std over 207 val subjects.

| Model | MAE(HU) body | PSNR body | SSIM body | MAE(HU) full | PSNR full | SSIM full | Hard Bone Dice | Hard Dice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet | 33.986 ± 14.318 | 26.419 ± 3.121 | 0.898 ± 0.036 | 37.347 ± 17.525 | 25.907 ± 3.543 | 0.887 ± 0.043 | 0.828 ± 0.065 | 0.740 ± 0.140 |
| koalAI | 40.280 ± 20.513 | 25.236 ± 3.637 | 0.879 ± 0.052 | 54.688 ± 18.091 | 25.052 ± 3.469 | 0.559 ± 0.061 | 0.777 ± 0.092 | 0.655 ± 0.129 |

## Track B: SynthRAD-native [-1024, 3000] (official ImageMetrics)

Official challenge metric vs raw full-HU CT (bone up to ~3000): MAE body-masked on raw HU; PSNR and MS-SSIM clip to [-1024,3000], data_range 4024. amix/unet/cWDM cap predictions at +1024 HU and MAISI at +1000 (their training clipped CT), so Track B penalizes them on dense cortical bone they structurally cannot represent (preprocessing choice, not raw model quality). koalAI was trained on full-range CT and is judged fairly here. Use Track A to compare models within [-1024,1024]; Track B for who delivers true bone HU (e.g. dose calculation). Mean ± std over 207 val subjects.

| Model | MAE(HU) | PSNR | MS-SSIM |
| --- | --- | --- | --- |
| unet | 91.462 ± 31.534 | 27.619 ± 2.781 | 0.900 ± 0.035 |
| koalAI | 104.527 ± 37.331 | 26.499 ± 2.991 | 0.863 ± 0.061 |

## Per-region (Track A)

Per-region all-class Hard Dice is region-confounded: trust Hard Bone Dice, not the all-class mean. The all-class Hard Dice averages over all 11 teacher organ classes, but a limited FOV (e.g. head/neck) anatomically contains only ~6. The GT seg (ct_seg.nii) carries spurious labels for absent torso organs (a head/neck case tagged with ~3,000 voxels of "breast implant" plus stray abdominal-cavity / pericardium / mediastinum), which synthesis correctly does not reproduce, so those classes score ~0 and drag the mean down (head/neck amix all-class 0.49 vs Bone 0.79). This is a metric + GT-label artifact, not a synthesis deficit, and hits every model equally. Hard Bone Dice (present in every region) is the trustworthy per-region Dice signal.

Worked example, head/neck subject 1HNC007 (amix prediction), per-class Hard Dice. The five spurious / out-of-FOV classes score ~0 and pull the 11-class mean to 0.394:

| class | GT vox | pred vox | Hard Dice |
| --- | --- | --- | --- |
| Brain | 300,196 | 313,412 | 0.963 |
| Bones | 212,082 | 226,037 | 0.810 |
| Muscle | 147,819 | 163,604 | 0.773 |
| Subcutaneous | 131,843 | 125,728 | 0.765 |
| Spinal cord | 1,792 | 667 | 0.501 |
| Gland | 4,345 | 5,895 | 0.447 |
| Abdominal cav | 1,300 | 1,272 | 0.000 |
| Breast implant | 2,975 | 116 | 0.043 |
| Thoracic cav | 396 | 25 | 0.033 |
| Pericardium | 1,915 | 0 | 0.000 |
| Mediastinum | 300 | 0 | 0.000 |
| mean (= Hard Dice) | - | - | 0.394 |

### Per-region body MAE (HU)

| Model | abdomen | brain | head_neck | pelvis | thorax |
| --- | --- | --- | --- | --- | --- |
| unet | 24.5 ± 6.6 | 51.2 ± 7.6 | 29.6 ± 7.0 | 40.0 ± 3.4 | 16.5 ± 4.0 |
| koalAI | 26.9 ± 10.3 | 58.7 ± 9.6 | 32.4 ± 9.7 | 59.3 ± 20.6 | 18.1 ± 6.7 |

### Per-region body PSNR

| Model | abdomen | brain | head_neck | pelvis | thorax |
| --- | --- | --- | --- | --- | --- |
| unet | 28.44 ± 1.69 | 22.97 ± 1.10 | 25.87 ± 1.41 | 25.38 ± 0.70 | 30.97 ± 1.41 |
| koalAI | 27.70 ± 2.29 | 21.74 ± 1.18 | 25.25 ± 1.88 | 22.94 ± 2.66 | 29.79 ± 2.24 |

### Per-region Hard Bone Dice

| Model | abdomen | brain | head_neck | pelvis | thorax |
| --- | --- | --- | --- | --- | --- |
| unet | 0.793 ± 0.039 | 0.861 ± 0.034 | 0.766 ± 0.044 | 0.931 ± 0.006 | 0.786 ± 0.026 |
| koalAI | 0.733 ± 0.071 | 0.813 ± 0.037 | 0.703 ± 0.084 | 0.920 ± 0.013 | 0.721 ± 0.058 |

### Per-region Hard Dice

| Model | abdomen | brain | head_neck | pelvis | thorax |
| --- | --- | --- | --- | --- | --- |
| unet | 0.834 ± 0.055 | 0.701 ± 0.057 | 0.471 ± 0.057 | 0.845 ± 0.051 | 0.825 ± 0.051 |
| koalAI | 0.720 ± 0.062 | 0.642 ± 0.064 | 0.414 ± 0.063 | 0.791 ± 0.055 | 0.685 ± 0.051 |

## Significance: paired Wilcoxon (body MAE)

Paired Wilcoxon on per-subject body_mae_hu (shared subjects). p<0.05 = significant difference; p>=0.05 = within noise (tie).

| vs | unet | koalAI |
| --- | --- | --- |
| unet | - | p=4.2e-19 |
| koalAI | p=4.2e-19 | - |

## Inference time

Per-volume inference. amix/unet/maisi re-measured with CUDA synchronization on 10 representative subjects (an earlier wall-clock without sync under-counted these fast models: unet ~11x, amix ~2.9x, maisi ~1.6x). Diffusion models are sync-invariant (GPU time dominated by multi-step sampling), so cWDM and MC-DDPM use full-207 generation means as-is. koalAI = end-to-end nnU-Net predict over all 207 (incl. per-region model load); amix/unet/maisi = sliding-window forward only (model load excluded). Both unet and koalAI here are pre-generated (n/a).
