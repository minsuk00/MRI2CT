# Where the U-Net sCT error lives, by ground-truth CADS label

**Source HTML:** _html/10_cads_error_decomposition.html
**Date:** 2026-06-20
**TL;DR:** Decomposing U-Net sCT error by GT CADS 35-label class: ~18% of "body" is external air a loose body mask wrongly includes (22% of total error, a preprocessing artifact). Aggregate error is a volume story (soft 47%, unlabeled air 23%, lungs 11% dominate; bone only 19% despite worst per-voxel MAE). Bone is located correctly (AUC 0.90) but undershot and blurred via L1 regression to the conditional mean. See report 16 (docs/16) for the same decomposition across all six models.

Plain 3D U-Net sCT (wandb `9xmodnhn`), 207 center-wise validation subjects (`full_eval_20260617`). Every body voxel is attributed to its ground-truth CADS 35-label class (no segmentation model, no HU-threshold tissue classes). All errors are full-range raw HU, error = sCT - GT, inside the body mask, accumulated as micro sums so the parts reconstruct the whole.

## 0. Method & consistency gate

- Attribution: each body voxel -> its GT CADS label (0-34). Groups: bone {7,27,28,29,30}; air-organs {airway 9, lungs 13}; soft = other labels 1-34; unlabeled = label 0 (CADS Background).
- Error: full-range raw HU |sCT - GT| (and signed bias = sCT - GT). The sCT is bounded to ≤~1024 by its sigmoid; GT is raw to ~3000.
- Micro & additive: per-label/group MAE = Σ|err| / Σvox, so the voxel-weighted parts sum to the body MAE.

**Gate PASS.** Σ|err| over all CADS labels / Σ body voxels = 73.8 HU = the body-voxel MAE (micro), and the group error-shares sum to 100.0%. The per-subject-averaged (macro) body MAE is 91.8 HU (= the leaderboard `synthrad_mae`); macro > micro because small-body subjects score higher and weigh equally.

## 1. The body mask is loose: ~18% of "body" is external air the mask wrongly includes

**Claim:** 18% of "body" voxels carry no CADS label (Background, mean -909 HU = air), and by a tight-body test 99% of it is EXTERNAL air outside the patient that a loose mask swept in (17.6% of body); only 0.15% of body is genuine internal unlabeled gas. That external air contributes 22% of the total body error (internal gas 0.5%), i.e. roughly a fifth of the reported body MAE comes from outside the patient.

**Why the model puts tissue there.** The input MR is zeroed only outside the body mask; inside the loose band it still carries low-level signal, and the saved sCT is the raw, unmasked network output (in `unet_baseline/validate.py` the body mask is applied only to the metrics, never to the written volume). So the U-Net translates that residual MR into HU. Pooled over 30 subjects, in voxels where the truth is air the sCT median climbs with MR intensity (r = 0.72), and ~10% of in-mask air voxels are lifted above -400 HU; the external-air MAE is 92 HU. The Background bias is near zero (-3) only because most of the band is correctly predicted air (MR ≈ 0); the error is the minority of band voxels where the MR is non-zero.

**Bottom line:** this ~22%-of-error term is largely a mask-tightness / preprocessing artifact, not a synthesis failure: the body mask is loose, the MR is not zeroed inside it, the sCT is saved unmasked, and the GT it is scored against is itself noisy there (streak artifacts). A tighter body mask (or tissue-restricted scoring) removes almost all of it. Genuine internal gas (bowel, lungs, airway) mostly carries its own CADS label, so it is not in this group.

## 2. What actually drives the body error (contribution, additive)

**Claim:** the body MAE is dominated by high-volume, non-bone regions: soft tissue (47%), unlabeled air (23%) and lungs (~11%) make up ~80%. Bone contributes only 19%, spread over 5 labels, despite being the worst per voxel. Severity != leverage.

| CADS group | % body vox | MAE | bias | % of body error |
| --- | --- | --- | --- | --- |
| bone (5 labels) | 7.8 | 181.7 | -94.1 | 19.1 |
| air-organs (airway+lung) | 7.9 | 102.7 | 75.0 | 10.9 |
| soft (other CADS) | 66.6 | 52.5 | 13.0 | 47.4 |
| unlabeled (CADS=0) | 17.8 | 93.8 | -2.9 | 22.6 |

**Metric-blindness to bone.** Bone is only 8% of body voxels, so the headline body-MAE / PSNR barely move with it; its 19% error share is swamped by soft (47%) and the loose-mask air (23%). The clinically critical error is nearly invisible to the aggregate metric; a tissue-restricted or bone-specific metric is needed to see it. This is the single most important caveat when reading MAE/PSNR leaderboards.

Top contributors:

| CADS label | % vox | MAE | bias | GT HU | pred HU | % error |
| --- | --- | --- | --- | --- | --- | --- |
| Background | 17.8 | 93.8 | -2.9 | -909.3 | -912.2 | 22.6 |
| Subcutaneous tissue | 22.7 | 62.3 | 8.0 | -78.5 | -70.5 | 19.2 |
| Lungs | 7.8 | 102.5 | 75.0 | -824.2 | -749.2 | 10.8 |
| Muscle | 18.1 | 34.1 | 4.6 | 10.2 | 14.9 | 8.4 |
| Bowel | 6.5 | 87.0 | 56.8 | -88.0 | -31.2 | 7.7 |
| Skull | 1.1 | 312.2 | -151.3 | 707.7 | 556.4 | 4.8 |
| Limb & girdle bones | 1.8 | 185.4 | -145.5 | 337.5 | 192.0 | 4.5 |
| Bone - other | 2.3 | 130.5 | -2.3 | 103.3 | 101.0 | 4.1 |
| Abdominal cavity | 5.6 | 46.2 | 22.2 | -64.8 | -42.6 | 3.5 |
| Spine | 1.7 | 146.0 | -91.1 | 280.1 | 189.1 | 3.4 |
| Thoracic cage | 0.8 | 211.4 | -170.9 | 305.4 | 134.5 | 2.3 |
| Liver | 3.6 | 38.6 | 15.9 | 50.6 | 66.5 | 1.9 |

**Note on grouping:** "air-organs" here is airway+lungs only (air-dominated, GT ≈ -800). Bowel (GT ≈ -88, a mixed gas/fluid/wall organ) is counted under "soft", consistent with the group decomposition, but it still over-predicts by +57 HU (the model fills its gas), like the other air-containing structures.

Full per-CADS-label table (MAE, bias, error share, micro, additive):

| CADS label | % vox | MAE | bias | GT HU | pred HU | % error | n subj |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Background | 17.8 | 93.8 | -2.9 | -909.3 | -912.2 | 22.6 | 207.0 |
| Subcutaneous tissue | 22.7 | 62.3 | 8.0 | -78.5 | -70.5 | 19.2 | 207.0 |
| Lungs | 7.8 | 102.5 | 75.0 | -824.2 | -749.2 | 10.8 | 82.0 |
| Muscle | 18.1 | 34.1 | 4.6 | 10.2 | 14.9 | 8.4 | 207.0 |
| Bowel | 6.5 | 87.0 | 56.8 | -88.0 | -31.2 | 7.7 | 109.0 |
| Skull | 1.1 | 312.2 | -151.3 | 707.7 | 556.4 | 4.8 | 94.0 |
| Limb & girdle bones | 1.8 | 185.4 | -145.5 | 337.5 | 192.0 | 4.5 | 109.0 |
| Bone - other | 2.3 | 130.5 | -2.3 | 103.3 | 101.0 | 4.1 | 207.0 |
| Abdominal cavity | 5.6 | 46.2 | 22.2 | -64.8 | -42.6 | 3.5 | 143.0 |
| Spine | 1.7 | 146.0 | -91.1 | 280.1 | 189.1 | 3.4 | 152.0 |
| Thoracic cage | 0.8 | 211.4 | -170.9 | 305.4 | 134.5 | 2.3 | 85.0 |
| Liver | 3.6 | 38.6 | 15.9 | 50.6 | 66.5 | 1.9 | 86.0 |
| Thoracic cavity | 1.2 | 91.6 | 6.2 | -176.0 | -169.7 | 1.5 | 150.0 |
| Heart | 1.6 | 53.9 | 28.8 | 14.7 | 43.5 | 1.2 | 104.0 |
| CSF | 0.4 | 139.5 | -49.3 | 230.5 | 181.2 | 0.8 | 91.0 |
| Stomach | 0.6 | 93.9 | 73.4 | -75.0 | -1.6 | 0.8 | 86.0 |
| Blood vessels | 0.9 | 54.1 | -23.3 | 57.0 | 33.7 | 0.7 | 207.0 |
| Gray matter | 1.8 | 24.5 | -2.9 | 51.5 | 48.6 | 0.6 | 91.0 |
| Spleen | 0.6 | 34.8 | -2.8 | 45.4 | 42.6 | 0.3 | 80.0 |
| Kidneys | 0.5 | 33.4 | -2.7 | 32.3 | 29.6 | 0.2 | 78.0 |
| Breast | 0.5 | 27.3 | 17.2 | -104.6 | -87.4 | 0.2 | 90.0 |
| Bladder | 0.2 | 61.5 | -51.1 | 11.4 | -39.7 | 0.1 | 34.0 |
| White matter | 1.2 | 7.9 | -0.8 | 32.6 | 31.8 | 0.1 | 91.0 |
| Spinal cord | 0.2 | 42.9 | 21.0 | 19.7 | 40.7 | 0.1 | 194.0 |
| Airway | 0.1 | 134.3 | 79.5 | -653.7 | -574.2 | 0.1 | 73.0 |
| Esophagus | 0.1 | 90.0 | -45.9 | -8.0 | -53.9 | 0.1 | 82.0 |
| Pancreas | 0.1 | 34.7 | 8.1 | 24.4 | 32.6 | 0.1 | 78.0 |
| Eyes & optic pathway | 0.0 | 73.8 | -7.9 | -62.7 | -70.6 | 0.0 | 79.0 |
| Prostate & seminal vesicle | 0.0 | 35.7 | -33.1 | 31.2 | -2.0 | 0.0 | 33.0 |
| Gallbladder | 0.0 | 48.8 | -5.6 | 14.9 | 9.3 | 0.0 | 44.0 |
| Face & oral soft tissue | 0.0 | 103.9 | -4.4 | -70.1 | -74.5 | 0.0 | 100.0 |
| Adrenals | 0.0 | 30.8 | 2.2 | -1.0 | 1.2 | 0.0 | 80.0 |
| Head & neck glands | 0.0 | 33.4 | -6.2 | 68.3 | 62.1 | 0.0 | 59.0 |
| Brain - other | 0.0 | 17.7 | 0.0 | 44.6 | 44.7 | 0.0 | 62.0 |
| Gland - other | 0.0 | 34.5 | -2.3 | 63.4 | 61.1 | 0.0 | 32.0 |

## 3. Per-voxel severity and direction (which way each tissue is wrong)

**Claim:** per voxel, bone is by far the worst and the only strongly under-predicted tissue (cortical -91 to -171 HU); air-filled organs are over-predicted (lungs +75, bowel +57); soft tissue is well-calibrated (+13). The model regresses both density extremes toward soft tissue.

Bone labels (dense cortical bone undershoots; "bone-other" is a soft-HU residual that does not):

| bone label | % vox | MAE | bias | GT HU | pred HU | % error |
| --- | --- | --- | --- | --- | --- | --- |
| Skull | 1.1 | 312.2 | -151.3 | 707.7 | 556.4 | 4.8 |
| Limb & girdle bones | 1.8 | 185.4 | -145.5 | 337.5 | 192.0 | 4.5 |
| Bone - other | 2.3 | 130.5 | -2.3 | 103.3 | 101.0 | 4.1 |
| Spine | 1.7 | 146.0 | -91.1 | 280.1 | 189.1 | 3.4 |
| Thoracic cage | 0.8 | 211.4 | -170.9 | 305.4 | 134.5 | 2.3 |

**One mechanism: regression to the conditional mean** (L1 loss -> conditional median). The rare high extreme (dense cortical bone) is pulled down and the low extreme (air) is pulled up, while the abundant middle (soft tissue) is rendered accurately. So the three findings above (bone undershoot, air overshoot, soft well-calibrated) are one effect, not three. The undershoot is concentrated in dense cortical bone (skull/ribs/limb/spine, GT 305-708 HU); the low-density "bone-other" filler does not undershoot. The sCT also saturates at the sigmoid's +1024 HU ceiling in 61/207 subjects, so it cannot represent the densest cortical bone at all.

## 4. Bone: localized, but undershot and blurred (CADS-bone, no segmenter)

**Claim:** the bone failure is not mislocalization. The GT-CADS bone is in the right place in the sCT (AUC 0.90 vs 0.92 ceiling). It is a density error in two parts: a bulk HU undershoot and a loss of edge sharpness (sCT bone edges 0.81x as sharp as real CT). Both are HU errors; the split is by spatial scale.

*Figure: the same GT-CADS-bone outline drawn on GT CT and sCT lands on bone in both (localization intact), but inside it the sCT is greyer (undershoot) and softer (blur). Error map: blue = undershoot at bone, red = overshoot at air.*

## 5. Conclusion

- **Mask looseness first.** ~18% of the scored "body" is unlabeled air, and 99% of it is external (outside the patient), contributing 22% of the body error. The model fills it because the MR is not zeroed inside the loose mask (sCT vs MR r = 0.72) and the sCT is saved unmasked, largely a preprocessing artifact. Any body-MAE comparison should use a tight mask or tissue-restricted error.
- **Aggregate error is a volume story.** Soft tissue (47%), unlabeled air (23%) and lungs (~11%) dominate; bone is only 19% despite the worst per-voxel error.
- **Bone is the clinically critical, per-voxel-worst error** (skull MAE 312; cortical undershoot -91 to -171 HU), and it is one-sided: dense bone is pulled down, air is pushed up, soft is fine (regression of both extremes toward soft tissue).
- **Bone is located, not misplaced** (AUC 0.90); the error is density: a bulk undershoot (recalibratable) plus edge blur (0.81x sharpness, not recalibratable). Fixing it needs HU calibration and sharper high-frequency detail, not relocation.
- **One root cause: regression to the conditional mean** (dense bone down, air up, soft accurate); the +1024 cap worsens the densest bone.
- **Metric-blindness:** because bone is ~8% of voxels, body-MAE/PSNR barely reflect it; the most clinically important error is nearly invisible to the headline metric. Report bone/tissue-restricted metrics, and use a tight body mask, for any meaningful comparison.

**Reproduce:** `cads_extract.py` -> `cads_analyze.py` -> `cads_mask_split.py` (tight-body external/internal split) -> `cads_loose.py` (MR->sCT mechanism) -> `cads_figures.py` -> `cads_report.py`; localization/blur from `verify_density.py` / `verify_blur.py` (all GT-CADS-label based, no babyseg).
