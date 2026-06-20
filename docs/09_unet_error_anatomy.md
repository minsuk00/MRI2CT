# Where the U-Net MR->CT Baseline Actually Fails

**Source HTML:** _html/04_unet_error_anatomy.html
**Date:** 2026-06-16
**TL;DR:** Two facts pointing opposite directions: (1) bone (especially cortical skull) is by far the worst-predicted tissue (MAE 318-274 HU vs ~56 HU soft; cortical undershot by 731 HU) and is information-limited (even uncapped diffusion can't fix it); (2) yet bone is NOT what caps the reported PSNR/MAE: oracle-substituting perfect bone raises PSNR only +1.29 dB (perfect cortical: +0.28 dB) because the metric clips at +/-1024 HU and is dominated by air/soft voxels (fixing air gives +2.78 dB). The standard leaderboard metric is blind to the one thing most broken and most clinically important. Part of the bone-failure series.

Error-anatomy of the plain 3D U-Net on 207 center-wise validation subjects (full_eval_20260609, ckpt 9xmodnhn ep799). Decompositions over the raw full-HU CT, the 35-label CADS segmentation, and an oracle "perfect-tissue" counterfactual. amix / MAISI / MC-DDPM included as contrast.

Validation: every recomputed metric matches the released pipeline exactly. Recomputed body-voxel MAE vs synthrad_mae and oracle base PSNR vs reported body_psnr agree to <0.001 (after correcting a RAS-vs-native orientation flip between the raw CT and the saved predictions).

## 1. Bone is the worst-predicted tissue, everywhere

Splitting every body voxel by ground-truth HU (air <-300, soft -300..200, bone >200): bone MAE is ~5x soft tissue in every region. The single largest per-voxel error source.

| region | MAE air | MAE soft | MAE bone | MAE all | body PSNR |
| --- | --- | --- | --- | --- | --- |
| brain | 151.2 | 65.3 | 318.3 | 123.9 | 22.9 |
| head_neck | 132.7 | 80.2 | 324.0 | 123.4 | 25.8 |
| thorax | 86.3 | 49.7 | 242.2 | 70.6 | 31.0 |
| abdomen | 111.9 | 46.7 | 257.0 | 68.1 | 28.4 |
| pelvis | 113.0 | 32.5 | 198.8 | 58.4 | 25.4 |

## 2. The failure is a systematic UNDERSHOOT, worst in the skull

Every skeletal structure is predicted too low (negative bias). The skull is worst (cortical, thin, ~30% of its voxels exceed 1024 HU). Inside true bone the network collapses toward the soft-tissue mean and essentially never emits dense cortical HU: classic regression-to-the-mean on a one-to-many target.

| bone structure | MAE (HU) | bias (HU) | GT mean HU | % vox >1024 | n subj |
| --- | --- | --- | --- | --- | --- |
| skull | 316.2 | -153.8 | 646.5 | 30.2 | 94.0 |
| limb_girdle | 218.4 | -172.9 | 332.8 | 3.1 | 109.0 |
| thoracic_cage | 218.2 | -179.8 | 306.9 | 0.1 | 85.0 |
| bone_other | 212.1 | 8.2 | 33.6 | 1.2 | 207.0 |
| spine | 168.9 | -104.9 | 315.1 | 0.8 | 141.0 |

*Figure: pooled HU inside true bone: GT spans to ~2800 HU; UNet piles up low (mean 542->329).*

## 3. It is information-limited, not a clipping artifact (the diffusion test)

UNet and amix are sigmoid-capped at +1024 HU, but MAISI and MC-DDPM are not. If the bone failure were merely the cap, the uncapped diffusion models would recover dense bone. They do not: MC-DDPM tops out ~1430 HU (vs real cortical ~2976) and still undershoots cortical bone by 750 HU. No model, capped or not, predicts dense bone. The bottleneck is missing information in the MR, not architecture.

| model | bone MAE | cortical MAE | cortical bias | pred bone MAX | pred bone mean |
| --- | --- | --- | --- | --- | --- |
| unet | 274.3 | 730.9 | -730.9 | 950.9 | 351.3 |
| amix | 273.2 | 739.0 | -739.0 | 929.9 | 348.8 |
| maisi | 344.8 | 900.4 | -900.4 | 992.4 | 261.2 |
| mcddpm | 299.7 | 765.7 | -750.4 | 1427.5 | 336.9 |

## 4. The failure is a density UNDERSHOOT, not a localization error

Separating the two bone failure modes: localization (does the model put bone in the right voxels? shape-Dice of pred>200 vs GT>200) and magnitude (where both agree a voxel is bone, how wrong is the HU?). Even on the agreed-bone intersection the error is 220 HU (dense interior worse at 330 HU vs 222 at the edge): a pure density undershoot. The undershoot is so severe in thin trabecular bone (thorax/abdomen ribs & vertebrae) that those voxels fall BELOW the 200-HU bone threshold entirely, appearing as a "missed-bone" localization error (~60% missed there) when it is really magnitude. In dense skull the model localizes bone well (Dice ~0.79) yet still undershoots its density.

## 5. The reported metric is blind to the bone error (oracle counterfactual)

Direct test of "is bone what limits the score?": overwrite the prediction with ground-truth HU inside one tissue and recompute the exact reported metrics. In the clipped PSNR metric, fixing bone barely moves the number (cortical: +0.28 dB); fixing air helps 10x more, because the metric clips cortical errors away and averages over far more air/soft voxels. In full-HU MAE (unclipped), fixing bone DOES help, especially in brain/HN.

| oracle scenario | all PSNR | brain PSNR | all full-HU MAE | brain full-HU MAE |
| --- | --- | --- | --- | --- |
| baseline | 26.40 | 22.92 | 91.82 | 123.92 |
| fix air | 29.18 | 25.53 | 57.58 | 78.07 |
| fix soft | 27.93 | 24.34 | 57.01 | 87.00 |
| fix bone (>200) | 27.69 | 24.38 | 69.04 | 82.77 |
| fix cortical (>1024) | 26.68 | 23.38 | 83.53 | 103.25 |
| fix skull | 26.82 | 23.99 | 80.18 | 92.09 |

This is the publishable hook: the thing most broken (cortical bone) and most relevant to dose/planning is precisely what the pixel metric cannot see. A model could halve its cortical-bone error and the leaderboard PSNR would barely move; concrete on-our-data evidence that pixel metrics mis-rank for the downstream task.

## 6. Region diagnoses: assumed causes, tested

Error-mass composition (share of each region's total abs-error): brain and head&neck carry a large bone share; abdomen/pelvis/thorax are air+soft dominated.

### Brain: it is the skull, not defacing

Hypothesis was MR/CT defacing mismatch. Tested: the face_oral structure contributes 0.0% of brain error mass (~205 voxels/subject), while the skull alone is 25.3% and all bone is ~33%. Brain's low PSNR is a cortical-skull + air-cavity problem; defacing is negligible.

### Pelvis: a modest OOD sequence shift, not a blow-up

Pelvis validation is 100% center C while training is center A only (documented T1->T2 sequence shift). Effect is real but mild: a soft-tissue undershoot bias of -9.4 HU and 2nd-lowest PSNR (25.4), but pelvis full-HU MAE (58) and bone MAE (199) are actually among the best regions. The T1/T2 shift nudges soft tissue; it is not the dominant error source.

### Head & neck: the organ-Dice "collapse" (0.47) is a metric artifact, not bad sCT

HN has the lowest organ-Dice, but a per-class breakdown (the eval's exact 12-class teacher) shows bulk tissues are fine: c1 (brain/soft) 0.68, c2 0.80, c5 (bone) 0.77, c11 0.86, comparable to other regions. The macro-average is dragged down by a few tiny structures (c3, c4, c8; ~1-4k voxels) scoring 0.03-0.36. Running the same teacher on the real GT CT as a ceiling proves these are not an sCT failure: the GT-CT ceiling for the whole region is only 0.48 and the sCT reaches 0.43 of it; the tiny classes (c3: sCT 0.032 vs GT-ceiling 0.031) are unsegmentable even with perfect CT. HN organ-Dice is bounded by the teacher/metric, not the synthesis.

## 7. Verdict: ranked drivers of UNet error

- Cortical-bone undershoot (information-limited). Largest per-voxel error, systematic -731 HU bias, worst in skull, unfixable by capacity or diffusion. Needs information the MR lacks (the case for an external prior / retrieval).
- Metric blindness. The reported PSNR/MAE clips bone and is air/soft-weighted, so it under-reports the bone failure (+0.28 dB for perfect cortical bone). Needs a downstream/decision-aware metric.
- Air/gas voxels dominate the aggregate metric by sheer count (biggest PSNR lever) though clinically trivial.
- OOD domain shift (pelvis/HN center C; pelvis T1->T2): modest soft-tissue bias, region-localized.
- Defacing: not a driver (0.0% of brain error mass), refuted.

Both top drivers point the same way as the methodology direction: cortical bone is missing information (an external CT prior / retrieval can inject what the MR cannot provide), and the failure is invisible to pixel metrics (motivating downstream-/bone-aware evaluation). The two are complementary halves of one paper.

## Reproducibility

Scripts in src/evaluate/error_anatomy/: extract.py (per-subject decomposition), oracle_fix.py (counterfactual), build_figures.py, examples.py, report.py. Data: evaluation_results/unet_error_analysis_20260616/ (summary.csv, structures.csv, oracle_fix.csv, key_stats.json). GT = raw ct.nii reoriented to canonical RAS to match the saved predictions; all 4 models scored against the same GT.
