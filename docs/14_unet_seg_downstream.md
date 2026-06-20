# U-Net MR->CT seg-downstream failure: localization vs density

**Source HTML:** _html/09_unet_seg_downstream.html
**Date:** 2026-06-19
**TL;DR:** Bone is the only failure; soft tissue and air are fine. Bone fails two independent ways: an HU undershoot (density, recalibratable, dominant for dose) and fine-structure blur (what a CNN segmenter sees, NOT fixable by intensity recalibration). Gross bone localization is intact (AUC 0.90 vs 0.92 ceiling).

Plain 3D U-Net baseline sCT (wandb `9xmodnhn`, center-wise split), 207 center-wise validation subjects (`full_eval_20260617`). A fixed CADS 35-label segmenter (BabyUNet `di54npq3`, same center-wise split, epoch 419) is run on the real CT and on the synthetic CT, scoring both against the ground-truth CADS segmentation. This separates two questions: does the sCT keep structures findable (localization, via Dice), and does it carry the right HU (density, via per-ROI bias)?

Sign convention: bias = pred - GT, so negative = undershoot.

## Bottom line

(This version corrects an earlier mechanistic overclaim; see section 6 verification.)

Bone is the failure; soft tissue and air are fine. But "bone" fails in two distinct ways that must not be conflated:

- **Density (HU magnitude).** Inside the true bone ROI the sCT undershoots HU by -106 HU (GT 405 -> pred 299); by raw HU value the undershoot is -182 HU. Gross localization is intact: sCT HU still separates bone from non-bone at AUC 0.90 vs the real-CT ceiling 0.92 (94% of above-chance separability). For an HU-threshold task, simply retuning the threshold recovers 74% of the bone-Dice gap. This is a calibration/scale error and is the clinically dominant one (dose/HU).
- **Fine bone structure (what a CNN segmenter sees).** The segmenter finds less bone in the sCT (bone-union Dice 0.83 -> 0.74). This is NOT the HU undershoot: the segmenter uses instance norm and is invariant to absolute HU, and a monotonic intensity recalibration (histogram-match sCT->GT, then re-segment) recovers -6% of that gap, i.e. nothing. The drop reflects structural degradation, measured to be blur: magnitude-matched sCT bone edges are 0.81x as sharp as real CT (section 6, Test D), a separate problem recalibration cannot fix.

So the model puts bone in roughly the right place; it both undershoots its HU and blurs its fine structure, and those two errors are independent (one fixable by recalibration, one not).

## 0. Method and correctness gates

- **Segmenter.** BabyUNet CADS model `di54npq3` (center-wise, epoch 419, `best.pth`), 35-output-channel U-Net. Loaded with `strict=True`; argmax gives labels 0-34, aligned 1:1 with GT CADS labels (verified).
- **Input normalization.** CT clipped to [-1024, 1024] HU, linearly mapped to [0, 1]. Verified equivalent to the training-time per-patch min-max stretch (Dice within 0.001; instance-norm cancels the difference).
- **Inference.** MONAI `sliding_window_inference`, 128^3 ROI, overlap 0.5, bf16. Real-CT input = dataset `ct.nii` (raw HU); sCT input = eval `sample.nii.gz`.
- **Dice.** 2|A∩B| / (|A|+|B|), each mask intersected with the body mask. Ceiling = Dice(babyseg(real CT), GT CADS); sCT = Dice(babyseg(sCT), GT CADS). A label absent in a subject is excluded (NaN), not scored 0.
- **HU bias / MAE (density).** Computed inside the GT CADS ROI (GT==label & body) on raw HU, segmenter-free.
- **Coarse groups.** bone = {skull, spine, thoracic cage, limb/girdle, bone-other}; air/lung = {airway, lungs}; soft = all other in-body labels.
- **Ceiling-correction.** The segmenter is imperfect even on real CT (small distilled student of full CADS), so real-CT Dice is the reachable ceiling; only the gap below it is attributed to the sCT.

Correctness gates: PASS.
- G1_bone_ceiling_dice = 0.833 (segmenter finds bone reliably on real CT, so low sCT bone Dice is sCT-attributable)
- G2_bone_hu_undershoot = -105.963 (bone HU undershoot is real)

## 1. Is each structure still findable in the synthetic CT?

| tissue group | ceiling (real CT) | synthetic CT | Dice drop |
| --- | --- | --- | --- |
| bone | 0.833 | 0.740 | 0.093 |
| soft | 0.969 | 0.945 | 0.024 |

Air is not a CADS class and is covered by HU in section 5. The drop is systematic, not an averaging artifact: the whole per-subject bone-union Dice distribution shifts down on the sCT.

## 2. Bone: localization or density?

Density error inside true ROIs is segmenter-independent (GT CADS ROI only). Each value is the per-subject mean HU error, averaged across subjects (macro, not voxel-weighted).

Full per-label table (signed bias and absolute MAE, with segmentability for reference):

| CADS label | group | Dice ceil | Dice sCT | Dice drop | GT HU | pred HU | HU bias | HU MAE | n subj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Brain - other |  | 0.00 | 0.00 | 0.00 | 42 | 42 | 1 | 20 | 28 |
| Adrenals |  | 0.00 | 0.00 | 0.00 | -3 | -2 | 1 | 35 | 79 |
| Spleen |  | 0.00 | 0.00 | 0.00 | 45 | 27 | -18 | 43 | 80 |
| Gallbladder |  | 0.00 | 0.00 | 0.00 | 14 | 18 | 4 | 47 | 42 |
| Face & oral soft tissue |  | 0.01 | 0.00 | 0.01 | -258 | -339 | -80 | 152 | 86 |
| Esophagus |  | 0.00 | 0.00 | 0.00 | -2 | -58 | -56 | 92 | 82 |
| Stomach |  | 0.03 | 0.02 | 0.01 | -76 | -6 | 70 | 100 | 86 |
| Pancreas |  | 0.04 | 0.03 | 0.02 | 21 | 25 | 4 | 37 | 78 |
| Gland - other |  | 0.02 | 0.03 | -0.01 | 66 | 59 | -7 | 38 | 27 |
| Prostate & seminal vesicle |  | 0.09 | 0.04 | 0.06 | 30 | -11 | -41 | 44 | 31 |
| Bladder |  | 0.49 | 0.04 | 0.45 | 13 | -41 | -54 | 69 | 34 |
| Head & neck glands |  | 0.04 | 0.05 | -0.01 | 79 | 67 | -12 | 44 | 39 |
| Breast |  | 0.09 | 0.08 | 0.01 | -102 | -133 | -31 | 102 | 80 |
| Blood vessels |  | 0.20 | 0.14 | 0.06 | 95 | 78 | -17 | 75 | 207 |
| Kidneys |  | 0.23 | 0.18 | 0.05 | 26 | 18 | -9 | 38 | 78 |
| Eyes & optic pathway |  | 0.17 | 0.19 | -0.02 | -62 | -71 | -9 | 77 | 77 |
| Spinal cord |  | 0.13 | 0.24 | -0.11 | 23 | 41 | 19 | 42 | 194 |
| CSF |  | 0.30 | 0.24 | 0.06 | 193 | 150 | -43 | 118 | 91 |
| Airway |  | 0.21 | 0.26 | -0.06 | -568 | -522 | 46 | 171 | 69 |
| Bone - other | bone | 0.43 | 0.27 | 0.16 | 34 | 42 | 8 | 212 | 207 |
| Spine | bone | 0.38 | 0.29 | 0.09 | 315 | 210 | -105 | 169 | 141 |
| Thoracic cavity |  | 0.36 | 0.33 | 0.03 | -221 | -184 | 38 | 125 | 111 |
| Thoracic cage | bone | 0.76 | 0.42 | 0.34 | 307 | 127 | -180 | 218 | 85 |
| Liver |  | 0.50 | 0.45 | 0.05 | 51 | 56 | 5 | 44 | 86 |
| Abdominal cavity |  | 0.54 | 0.46 | 0.08 | -63 | -53 | 10 | 59 | 134 |
| Bowel |  | 0.73 | 0.50 | 0.23 | -109 | -39 | 70 | 106 | 109 |
| Muscle |  | 0.56 | 0.51 | 0.05 | -10 | -0 | 10 | 85 | 207 |
| White matter |  | 0.56 | 0.55 | 0.01 | 34 | 31 | -3 | 12 | 91 |
| Limb & girdle bones | bone | 0.69 | 0.56 | 0.13 | 333 | 160 | -173 | 218 | 109 |
| Skull | bone | 0.68 | 0.60 | 0.08 | 646 | 493 | -154 | 316 | 94 |
| Heart |  | 0.40 | 0.60 | -0.20 | -41 | -43 | -2 | 80 | 104 |
| Gray matter |  | 0.69 | 0.66 | 0.03 | 48 | 43 | -6 | 25 | 91 |
| Subcutaneous tissue |  | 0.92 | 0.88 | 0.04 | -48 | -32 | 16 | 89 | 207 |
| Lungs |  | 0.95 | 0.93 | 0.02 | -789 | -695 | 94 | 123 | 82 |

**Note on "bone-other".** The CADS bone union is {skull, spine, thoracic cage, limb/girdle, bone-other}. "Bone-other" is a residual class with mean GT HU ~34 (soft-tissue range, present in all 207 subjects), so it barely undershoots (bias +8) but has large two-sided error (MAE 212). The undershoot is concentrated in dense cortical bone: skull -154, thoracic cage -180, limb/girdle -173, spine -105 HU (GT 307-646). The bone-union mean (-106) is diluted by bone-other.

Across regions, bone Dice drop and HU bias co-vary, but correlation is not the mechanism (section 6 shows the segmenter drop is not caused by HU magnitude). Of true (GT) bone voxels, the segmenter keeps 90% as bone on real CT but only 79% on the sCT, reassigning the rest mostly to soft tissue/muscle.

**Two effects, do not conflate them.** The HU undershoot is real and segmenter-free. The segmenter bone-Dice drop and relabeling are also real, but as section 6 proves they are NOT caused by the undershoot (the segmenter is intensity-invariant). An earlier version claimed the undershoot "pushes bone below the segmenter's bone appearance, so it relabels"; that causal link is refuted in section 6. The segmenter drop instead reflects degraded fine bone structure (blur / lost cortical detail).

## 3. Per region

| region | bone ceiling | bone sCT | bone Dice drop | bone HU bias | bone MAE |
| --- | --- | --- | --- | --- | --- |
| brain | 0.81 | 0.73 | 0.08 | -165.07 | 323.69 |
| head_neck | 0.53 | 0.44 | 0.08 | -78.61 | 256.45 |
| thorax | 0.91 | 0.79 | 0.12 | -84.79 | 163.09 |
| abdomen | 0.92 | 0.79 | 0.13 | -81.44 | 160.27 |
| pelvis | 0.97 | 0.93 | 0.04 | -82.71 | 144.55 |

## 5. Air / soft / bone by HU value (segmenter-free, density view)

The CADS segmenter has no "air" class, so true air (gas, ~-1000 HU) is defined by HU. Every body voxel is classified by HU value alone: air < -300, soft -300..150, bone > 150, then GT CT vs sCT compared directly (no segmentation).

| HU tissue | GT HU | pred HU | HU bias | HU MAE | vox % |
| --- | --- | --- | --- | --- | --- |
| air | -870 | -823 | 47 | 123 | 28 |
| soft | -10 | 4 | 14 | 54 | 63 |
| bone | 517 | 336 | -182 | 253 | 9 |

**By HU, the failure is bone, not air.** Air is handled well: true-air voxels predict -823 HU vs GT -870 (mild +47 overshoot), 94% of true-air voxels still read as air (6% leak to soft). Soft tissue is well calibrated (bias +14). Bone is the outlier: HU-defined bone (>150 HU) undershoots by -182 HU (GT 517 -> pred 336), so 35% of true-bone voxels drop below 150 HU and read as soft tissue, with a visible pile-up at the sigmoid's +1024 HU cap. The model regresses the dense extreme toward soft-tissue density while leaving air and soft essentially correct.

## 6. Verification: density or localization? (four falsification tests)

The segmenter is HU-driven, so the localization-vs-density question is tested with experiments designed to break the claim, including two that do not use the segmenter at all.

### Test A - is bone still spatially separable by HU? (segmenter-free, 207 subjects)
If bone is in the right place but too low in value, sCT HU should still rank true-bone above non-bone nearly as well as real CT. It does: AUC 0.901 (sCT) vs 0.925 (real CT) = 94% of above-chance separability retained. Gross bone localization is intact.

### Test B - does pure recalibration fix an HU-threshold bone task? (segmenter-free)
Thresholding HU at 150 gives bone Dice 0.59 on sCT vs 0.67 on real CT; retuning the single threshold (best ~184 HU) lifts sCT to 0.65, recovering 74% of the gap. For a threshold/HU task the error is largely a recalibratable scale problem.

### Test C - causal oracle: does fixing intensity recover the CNN segmenter? (GPU, 30 subjects)
A monotonic histogram-match of the sCT to the GT CT (cannot move anything spatially, only remaps values; verified to lift bone HU, e.g. 128->161 on one case), then re-run the segmenter. Bone Dice goes 0.733 -> 0.727 (ceiling 0.830): recovery -6%, essentially none. The segmenter uses instance norm and is invariant to absolute HU (identical Dice under different intensity scalings). So the segmenter's bone-Dice drop is NOT caused by the HU undershoot; it is structural, which recalibration cannot repair.

### Test D - is the structural defect actually blur? (measured, 25 subjects)
Histogram-match the sCT to the GT (removing the magnitude difference) and measure gradient magnitude on the GT-bone surface. If magnitude-matched sCT still has softer bone edges, that is genuine blur. It does: bone-edge sharpness is 0.81x the real CT's overall. The deficit tracks the Dice drop region-wise: pelvis ~1.00 (no blur, smallest Dice drop 0.04) while thorax/abdomen/brain sit at 0.75-0.76 (blur, larger drops). The segmenter loss is lost high-frequency bone detail (cortical edges), confirmed not inferred.

**Correction.** An earlier version concluded "bone fails purely by density; the segmenter relabels because HU is low." Test C refutes the causal link (perfect intensity fix recovers -6%). Corrected reading: two independent defects, an HU undershoot (density, recalibratable, dominant for dose/threshold tasks) and a fine-structure/texture degradation (what the CNN sees, not recalibratable).

## 7. Conclusion

- Soft tissue and air are fine (soft Dice drop 0.02; air kept 94% as air, +47 HU; soft +14 HU).
- Gross bone localization is intact. sCT HU separates bone from non-bone at AUC 0.90 vs 0.92 ceiling (94% retained).
- Bone fails in two independent ways:
  - (a) Density: HU undershoot -106 HU (bone ROI) / -182 HU (HU>150); retuning a threshold recovers 74% of an HU-task's gap, a scale/calibration error and clinically dominant.
  - (b) Fine structure: the CNN segmenter loses bone (Dice 0.83->0.74); a pure intensity fix recovers only -6%, and magnitude-matched bone edges are 0.81x as sharp as real CT, a sharpness/blur degradation, not an HU effect.
- Implication: for dose/HU accuracy, fix the bone calibration (magnitude). For a segmenter or any edge/texture-dependent downstream use, calibration alone will not help; the sCT's bone structure needs to be sharper. The two require different interventions.

## Reproducibility

Scripts in `src/evaluate/unet_failure/`: `seg_infer.py` -> `seg_extract.py` -> `seg_aggregate.py` -> `seg_figures.py`; HU view `hu_tissue.py`; verification `verify_density.py`, `verify_recalib_reseg.py`, `verify_blur.py`, `verify_figure.py`; assemble `seg_report.py`.
