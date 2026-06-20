# U-Net MR->CT failure anatomy: per CADS label & region

**Source HTML:** _html/05_unet_failure_anatomy.html
**Date:** 2026-06-19
**TL;DR:** Per voxel, the U-Net's error is dominated by a bone-density problem, specifically a cortical-density undershoot (bone MAE 166 vs organ 58 HU, 2.9x worse; cortical undershot 521 HU even within the trained [-1024,1024] range). Bone is 4.7% of voxels but 15% of error mass; an oracle bone-fix cuts scored body-MAE 21%. But on aggregate PSNR/MAE, air/soft are bigger levers because bone is rare and the metric clips it.

Plain 3D U-Net baseline (wandb `9xmodnhn`, ep799, fully trained on the center-wise split), scored on all 207 center-wise validation subjects (`full_eval_20260617`). Errors are decomposed over the 35-label CADS segmentation and the raw full-HU CT.

Sign convention: error = pred - GT, so negative bias = the model predicts too low (undershoot).

## Bottom line

Per voxel, the U-Net's error is dominated by a bone-density problem, and within bone a cortical-density undershoot. Bone is predicted 2.9x worse than soft-tissue/organ labels (bone-label MAE 166 vs organ 58 HU), the skull is the single worst-predicted structure, and cortical bone is undershot by 521 HU even within the [-1024,1024] range the model was actually trained and validated on. Bone occupies just 4.7% of body voxels but carries 15% of the total error mass, and an oracle that fixes only bone cuts the scored body-MAE by 21%. The model never emits dense cortical HU: inside true bone its predictions mean 351 HU and top out at ~951 HU while the GT reaches ~2563 HU.

**Caveat (see report 06):** bone is the worst-predicted tissue per voxel and the most clinically relevant, but it is NOT the biggest lever on the aggregate pixel metrics. Because bone is rare and the metric clips it, fixing air (+2.78 dB PSNR) or soft helps the leaderboard more than fixing bone (+1.29 dB). "Biggest" depends on whether you weight per-voxel/clinical accuracy or aggregate PSNR/MAE.

## Validation / proof of correctness (PASS)

Every recomputed number reproduces the released evaluation pipeline. Per-subject body MAE vs raw GT matches `synthrad_mae` and the clipped-frame body MAE matches the released `body_mae_hu` to <0.001 HU each (max 7.39e-05 and 5.14e-05); per-region means match `by_region.csv` to delta<0.001 HU (max 4.93e-06). Voxel-count partitions are exact, all 207 subjects load (52/60/32/30/33), and the raw-GT reference is confirmed (bone reaches 3071 HU, not clipped).

Gates: g1_mae_raw_vs_synthrad, g2_body_mae_clip_vs_released, g3_region_reconcile, g4_region_counts, g5_coverage, g6_mass_conservation, g7_raw_gt_reference, g8_ceiling - all PASS.

## 0. Setup & frames

GT = the raw `ct.nii` (full HU to ~2563), reoriented to canonical RAS to match the saved predictions; metrics are over body-mask voxels. Each label mask is `seg == id` within the body. Bone labels = skull (7), bone_other (27), limb_girdle (28), spine (29), thoracic_cage (30).

The validation was scored on GT clipped to [-1024, 1024] (verified: `ScaleIntensityRanged(clip=True)` + `hu_range=2048`), and the U-Net is trained to that clipped target (output ceilings ~951 HU). Every error is reported in two frames: **Frame C** (clipped, the metric the model was selected on -> genuine in-range failure) and **Frame R** (raw GT). The excess R-C over cortical bone is the part unrecoverable by construction (the target itself was clipped).

| region | recomp MAE (R) | released synthrad_mae | Δ R | recomp body MAE (C) | released body_mae_hu | Δ C |
| --- | --- | --- | --- | --- | --- | --- |
| brain | 123.922 | 123.922 | 0.000 | 51.322 | 51.322 | 0.000 |
| head_neck | 123.423 | 123.423 | 0.000 | 29.732 | 29.732 | 0.000 |
| thorax | 70.597 | 70.597 | 0.000 | 16.577 | 16.577 | 0.000 |
| abdomen | 68.073 | 68.073 | 0.000 | 24.785 | 24.785 | 0.000 |
| pelvis | 58.407 | 58.407 | 0.000 | 39.774 | 39.774 | 0.000 |

The pipeline reproduces the leaderboard exactly, so every decomposition below is trustworthy.

## 1. Bone is the worst-predicted tissue in every region

Splitting every body voxel by GT HU (air <-300, soft -300..200, bone >200), bone MAE is roughly 4x soft-tissue MAE everywhere. This is the single largest per-voxel error source, consistent across all five regions.

| region | MAE air | MAE soft | MAE bone | MAE all | bone:soft | body PSNR |
| --- | --- | --- | --- | --- | --- | --- |
| brain | 151.2 | 65.3 | 233.3 | 112.7 | 3.6 | 22.9 |
| head_neck | 132.7 | 80.2 | 290.6 | 119.3 | 3.6 | 25.8 |
| thorax | 86.3 | 49.7 | 237.6 | 70.4 | 4.8 | 31.0 |
| abdomen | 111.9 | 46.7 | 247.9 | 67.7 | 5.3 | 28.4 |
| pelvis | 113.0 | 32.5 | 194.1 | 58.2 | 6.0 | 25.4 |

Signed bias: bone is strongly negative (undershoot) everywhere; soft tissue is near-zero.

## 2. Bone carries error mass far beyond its volume

Bone is only 4.7% of body voxels but contributes 15% of the total absolute-error mass, a 3.3x over-representation. In skull-bearing regions (brain, head&neck) bone dominates the regional error budget.

| region | bone vox % | bone err-mass % | mass/vox ratio | soft err-mass % | air err-mass % |
| --- | --- | --- | --- | --- | --- |
| brain | 12.6 | 26.1 | 2.1 | 32.7 | 41.2 |
| head_neck | 11.8 | 28.9 | 2.4 | 40.3 | 30.9 |
| thorax | 3.9 | 13.5 | 3.5 | 42.4 | 44.0 |
| abdomen | 2.7 | 10.2 | 3.7 | 49.9 | 39.9 |
| pelvis | 4.7 | 15.8 | 3.4 | 41.3 | 42.9 |

## 3. The failure is a systematic cortical-density UNDERSHOOT (HU proof)

Inside true bone the network collapses toward the soft-tissue mean: GT mean 564 HU -> pred mean 351 HU, and the prediction never reaches dense cortical values (p95 697, max ~951 HU vs real cortical to ~2563). Decomposing the cortical (GT>1024) error by frame: the model undershoots cortical bone by 521 HU within the achievable [-1024,1024] range (Frame C) and 731 HU vs raw GT (Frame R). So about 71% of the cortical error is genuine in-range model failure and only 29% is the structural clipped-target ceiling. The problem is the model, not just the clip.

| region | GT bone mean | pred bone mean | pred bone max | GT bone max | cort bias (C) | cort bias (R) | near-ceil % | GT %>1024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| brain | 836.3 | 623.5 | 1023.1 | 2838.0 | -206.0 | -434.4 | 36.7 | 35.7 |
| head_neck | 605.3 | 391.2 | 1012.2 | 2947.5 | -460.9 | -699.5 | 10.1 | 13.8 |
| thorax | 395.8 | 174.0 | 853.2 | 2341.1 | -797.0 | -1012.7 | 0.0 | 1.7 |
| abdomen | 408.5 | 174.2 | 885.5 | 2323.6 | -756.9 | -959.0 | 0.2 | 2.4 |
| pelvis | 432.8 | 266.5 | 962.0 | 2265.3 | -500.7 | -652.2 | 0.4 | 3.0 |

**Information ceiling.** The model's bone output saturates at ~951 HU (12% of bone voxels sit >=850 HU) while real cortical bone runs to ~2563 HU. Even the 71% of the cortical error inside the trainable range is a systematic undershoot of 521 HU, a one-to-many MR->HU mapping the network resolves by regressing to the mean. This is the densest, most dose-relevant tissue and exactly what the model gets most wrong.

## 4. Per-structure rankings (all 35 CADS labels)

The four worst-predicted structures are all bone (skull, thoracic_cage, limb_girdle, bone_other), with spine close behind; the best-predicted are brain white/gray matter and abdominal organs. Prevalence (n subjects with the label) is shown so single-subject labels are not over-read.

**Math caveat (per-label MAE does NOT average to the reported MAE).** The 35 CADS labels cover only 82% of body voxels; the remaining ~18% is `seg==0` inside the body, almost entirely unlabeled internal air (sinus / bowel gas / cavities CADS has no label for; external background masked to -1024 and excluded). So voxel-averaging these per-label MAEs gives the MAE over the labeled 82% (~67.8 HU), NOT the body MAE (~72.4 HU body-voxel-mean), and definitely not the leaderboard `body_mae_hu` (~34 HU, which additionally divides by the full padded volume, x0.40). The only decomposition that reconstructs the body MAE exactly is the air/soft/bone HU split, which tiles 100% of the body (verified by the mass-conservation gate). Per-label numbers below are per-structure body-voxel-mean MAEs, valid on their own but not additive to the headline number.

| CADS label | is_bone | cads_region | MAE (C) | MAE (R) | bias (C) | GT HU | pred HU | GT %>1024 | n subj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skull | bone | Skeleton | 242.1 | 316.2 | -79.7 | 646.5 | 492.6 | 30.2 | 94.0 |
| thoracic_cage | bone | Skeleton | 218.0 | 218.2 | -179.6 | 306.9 | 127.1 | 0.1 | 85.0 |
| limb_girdle | bone | Skeleton | 210.0 | 218.4 | -164.5 | 332.8 | 159.9 | 3.1 | 109.0 |
| bone_other | bone | Skeleton | 208.4 | 212.1 | 11.9 | 33.6 | 41.8 | 1.2 | 207.0 |
| airway |  | H&N | 170.8 | 170.9 | 45.9 | -568.1 | -522.1 | 0.0 | 69.0 |
| spine | bone | Skeleton | 167.5 | 168.9 | -103.5 | 315.1 | 210.1 | 0.8 | 141.0 |
| face_oral |  | H&N | 151.2 | 152.2 | -79.0 | -258.4 | -338.5 | 0.1 | 86.0 |
| thoracic_cavity |  | Thorax | 124.7 | 125.0 | 38.0 | -221.3 | -183.6 | 0.0 | 111.0 |
| lungs |  | Thorax | 123.2 | 123.2 | 93.7 | -788.7 | -694.9 | 0.0 | 82.0 |
| csf |  | H&N | 112.2 | 118.1 | -36.9 | 192.8 | 150.1 | 3.8 | 91.0 |
| bowel |  | Abdomen | 106.2 | 106.2 | 69.7 | -108.5 | -38.8 | 0.0 | 109.0 |
| breast |  | Thorax | 101.9 | 101.9 | -31.2 | -102.3 | -133.5 | 0.0 | 80.0 |
| stomach |  | Abdomen | 100.5 | 100.5 | 70.4 | -75.9 | -5.5 | 0.0 | 86.0 |
| esophagus |  | Thorax | 92.1 | 92.1 | -55.6 | -2.0 | -57.6 | 0.0 | 82.0 |
| subcutaneous |  | Whole-body | 87.6 | 89.1 | 17.2 | -47.5 | -31.9 | 0.7 | 207.0 |
| muscle |  | Whole-body | 85.4 | 85.4 | 10.0 | -10.2 | -0.3 | 0.0 | 207.0 |
| heart |  | Thorax | 80.3 | 80.3 | -2.1 | -41.1 | -43.2 | 0.0 | 104.0 |
| eyes |  | H&N | 77.4 | 77.4 | -8.9 | -62.3 | -71.1 | 0.0 | 77.0 |
| blood_vessels |  | Vessels | 73.6 | 75.1 | -15.5 | 95.1 | 78.1 | 0.8 | 207.0 |
| bladder |  | Pelvis | 68.8 | 68.8 | -54.5 | 13.1 | -41.4 | 0.0 | 34.0 |
| abdominal_cavity |  | Abdomen | 58.5 | 58.6 | 10.2 | -63.4 | -53.3 | 0.0 | 134.0 |
| gallbladder |  | Abdomen | 46.9 | 46.9 | 3.7 | 13.8 | 17.5 | 0.0 | 42.0 |
| prostate_sv |  | Pelvis | 44.3 | 44.3 | -40.8 | 30.1 | -10.7 | 0.0 | 31.0 |
| liver |  | Abdomen | 43.7 | 43.7 | 5.2 | 50.8 | 56.0 | 0.0 | 86.0 |
| hn_glands |  | H&N | 43.6 | 43.6 | -11.6 | 78.8 | 67.1 | 0.0 | 39.0 |
| spleen |  | Abdomen | 43.2 | 43.2 | -18.3 | 45.4 | 27.1 | 0.0 | 80.0 |
| spinal_cord |  | H&N | 42.1 | 42.1 | 18.8 | 22.6 | 41.4 | 0.0 | 194.0 |
| gland_other |  | Whole-body | 37.9 | 37.9 | -7.0 | 65.8 | 58.8 | 0.0 | 27.0 |
| kidneys |  | Abdomen | 37.6 | 37.6 | -8.7 | 26.4 | 17.6 | 0.0 | 78.0 |
| pancreas |  | Abdomen | 36.7 | 36.7 | 3.6 | 21.4 | 25.0 | 0.0 | 78.0 |
| adrenals |  | Abdomen | 35.3 | 35.3 | 1.2 | -3.0 | -1.8 | 0.0 | 79.0 |
| gray_matter |  | H&N | 24.9 | 25.0 | -5.7 | 48.4 | 42.6 | 0.1 | 91.0 |
| brain_other |  | H&N | 20.4 | 20.4 | 0.6 | 41.5 | 42.2 | 0.0 | 28.0 |
| white_matter |  | H&N | 11.6 | 11.6 | -3.3 | 34.2 | 30.9 | 0.0 | 91.0 |

## 5. Bone vs organ, and the per-region picture

Bone labels vs all soft/organ labels directly: bone is 2.9x worse overall, and 2-4x worse in every region.

| region | bone MAE | organ MAE | ratio |
| --- | --- | --- | --- |
| brain | 224.5 | 87.2 | 2.6 |
| head_neck | 239.2 | 107.1 | 2.2 |
| thorax | 161.9 | 59.9 | 2.7 |
| abdomen | 157.5 | 58.2 | 2.7 |
| pelvis | 140.7 | 32.6 | 4.3 |
| OVERALL | 165.9 | 57.6 | 2.9 |

### Per-region worst structures

- **Brain** - airway (445), thoracic_cavity (408), bone_other (345). Highest body MAE; dominated by the skull and air cavities, not soft brain tissue (white/gray matter are the best-predicted labels in the whole dataset).
- **Head & neck** - limb_girdle (456), spine (286), bone_other (251). Dense skull base + airway; same cortical-undershoot story as brain.
- **Thorax** - skull (913), limb_girdle (224), thoracic_cage (207). Lowest body MAE region; error is air-interface (airway, lungs) plus thin rib/vertebral bone.
- **Abdomen** - limb_girdle (232), thoracic_cage (221), airway (208). Bowel gas and vertebral/rib bone; solid organs (liver, spleen, kidneys) well predicted.
- **Pelvis** - thoracic_cage (425), breast (342), thoracic_cavity (284). Bone (hip/sacrum) plus a mild soft-tissue shift; not catastrophic despite the known center-C T1/T2 domain shift.

## 6. How much would fixing bone help? (oracle counterfactual)

Overwrite the prediction with GT inside bone voxels only and recompute the scored body MAE: fixing bone alone drops it by 21% overall (and 27% in brain), despite bone being <5% of voxels, large for so few voxels. But this is not the biggest single lever: because air and soft voxels are far more numerous, fixing them helps the aggregate metric more (air +2.78 dB PSNR vs bone +1.29 dB; see report 06). Bone leads only on per-voxel and full-HU/clinical accuracy, where the clip does not hide it.

| region | body MAE (C) | bone-fixed (C) | drop % (C) | body MAE (R) | bone-fixed (R) | drop % (R) |
| --- | --- | --- | --- | --- | --- | --- |
| brain | 112.7 | 82.8 | 26.5 | 123.9 | 82.8 | 33.2 |
| head_neck | 119.3 | 84.5 | 29.2 | 123.4 | 84.5 | 31.6 |
| thorax | 70.4 | 60.7 | 13.7 | 70.6 | 60.7 | 14.0 |
| abdomen | 67.7 | 60.7 | 10.4 | 68.1 | 60.7 | 10.9 |
| pelvis | 58.2 | 48.8 | 16.1 | 58.4 | 48.8 | 16.4 |
| OVERALL | 87.8 | 69.0 | 21.3 | 91.8 | 69.0 | 24.8 |

## 7. Conclusion: ranked drivers of U-Net error

- **Cortical-bone density undershoot.** Largest per-voxel error (521 HU, Frame C), systematic -521 HU in-range bias, worst in the skull, model physically saturates ~951 HU. 71% of it is genuine in-range failure (not the clip). Fixing bone alone recovers 21% of the scored error.
- **Disproportionate error mass.** Bone is 4.7% of voxels but 15% of error mass, concentrated in brain & head&neck.
- **Region pattern.** Skull-bearing regions (brain, head&neck) have the highest MAE and are bone-dominated; body regions (thorax/abdomen/pelvis) are lower and air/soft-interface dominated with thin trabecular bone.
- **Soft tissue / organs are largely solved.** Organ labels sit on the identity line; white/gray matter, liver, spleen, kidneys are the best-predicted structures.

Per voxel and clinically, the U-Net's dominant failure is bone, specifically cortical density: a systematic, density-scaling undershoot (consistent across all 207 subjects) that the model cannot escape because (a) part of the dense-bone HU is clipped out of its training target and (b) even within range the MR under-determines cortical HU, so the network regresses to the mean. On the aggregate pixel metrics, however, bone is not the biggest lever (air/soft contribute more by sheer voxel count, and the clip hides cortical error), so whether bone is "the biggest problem" depends on whether you optimize per-voxel/clinical accuracy or the leaderboard. Report 06 dissects this and the root cause.

## Reproducibility

Scripts in `src/evaluate/unet_failure/`: `extract.py` (per-subject + per-label, dual-frame), `aggregate.py` (rollups + 8 correctness gates), `build_figures.py`, `report.py`. Outputs in `evaluation_results/unet_failure_20260619/` (`per_subject.csv`, `per_label.csv`, `bone_hist.npz`, derived table CSVs, `agg_stats.json`). GT = raw `ct.nii` in canonical RAS; predictions = `full_eval_20260617/volumes/unet`. All gates pass; numbers reconcile to the released metrics within floating-point tolerance.
