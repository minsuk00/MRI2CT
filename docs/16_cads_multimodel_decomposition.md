# Where the sCT error lives, by ground-truth CADS label: all six models

**Source HTML:** _html/11_cads_multimodel_decomposition.html
**Date:** 2026-06-20
**TL;DR:** Report 10's (docs/15) U-Net CADS error decomposition replicated across all six MR->CT models (U-Net, Anatomix, MAISI, cWDM, MC-DDPM, koalAI). The failure structure is shared by all: severity != leverage, ~20% external loose-mask air, regression-to-the-mean (bone down/air up/soft flat), and located-but-undershot-and-blurred bone. The only genuine difference is the output ceiling: uncapped models (MC-DDPM, koalAI) reduce the >1024 HU bone undershoot, but overall bone MAE barely moves because bone is information-limited, not clipping-limited.

Six MR->CT models on the same 207 center-wise validation subjects (`full_eval_20260617`). Every body voxel is attributed to its ground-truth CADS 35-label class (no segmentation model, no HU-threshold tissue classes). Errors are full-range raw HU, error = sCT - GT, inside the body mask, accumulated as micro sums. Replicates report 10 (docs/15, U-Net only) for every model with identical code.

## 0. Method, output ceilings, and consistency gate

- Attribution: each body voxel -> its GT CADS label. Groups: bone {7,27,28,29,30}; air-organs {airway 9, lungs 13}; soft = other labels 1-34; unlabeled = label 0 (CADS Background).
- Micro & additive: per-label/group MAE = Σ|err| / Σvox, so the parts reconstruct the body MAE.
- Output ceiling differs by model and is the crux of the bone comparison: U-Net/Anatomix use a sigmoid that saturates near 1024 HU, MAISI clips at 1000, cWDM clips at 1024, while MC-DDPM and koalAI have no hard ceiling and do emit voxels above 1024 HU.

| model | ceiling behaviour | max sCT bone HU | mean sCT HU @ GT>1024 |
| --- | --- | --- | --- |
| U-Net | sigmoid ~1024 | 1030 | 765 |
| Anatomix | sigmoid ~1024 | 1030 | 723 |
| MAISI | clip 1000 | 1010 | 607 |
| cWDM | clip 1024 | 1030 | 675 |
| MC-DDPM | none (>1024) | 1590 | 867 |
| koalAI | none (>1024) | 1590 | 976 |

**Consistency gate.** For every model, Σ|err| over all CADS labels reconstructs the body-voxel count exactly and the group error-shares sum to 100%. Whole-body MAE per model (macro = the leaderboard `synthrad_mae`):

| model | micro MAE | macro MAE (synthrad_mae) |
| --- | --- | --- |
| U-Net | 73.8 | 91.8 |
| Anatomix | 76.4 | 95.2 |
| MAISI | 100.4 | 125.7 |
| cWDM | 114.4 | 123.9 |
| MC-DDPM | 90.8 | 104.5 |
| koalAI | 85.0 | 104.5 |

Macro (per-subject mean) exceeds micro (voxel-pooled) for all models because smaller-body subjects score higher and weigh equally.

## 1. Severity vs leverage holds for every model

**Claim (report 10, U-Net):** bone is the worst tissue per voxel but a small slice of total error (17-19% across models), because it is only a few percent of voxels. Soft tissue + unlabeled air dominate the error mass by volume. This pattern is the same across all six models.

Group error-share matrix (% of body error):

| model | bone | air-organs | soft | unlabeled |
| --- | --- | --- | --- | --- |
| U-Net | 19.1 | 10.9 | 47.4 | 22.6 |
| Anatomix | 18.3 | 11.1 | 46.3 | 24.3 |
| MAISI | 18.7 | 7.5 | 54.6 | 19.2 |
| cWDM | 16.5 | 7.8 | 60.3 | 15.3 |
| MC-DDPM | 17.4 | 7.4 | 54.0 | 21.2 |
| koalAI | 16.6 | 8.5 | 45.6 | 29.3 |

## 2. External loose-mask air contributes a fifth of the error for every model

**Claim:** a loose body mask sweeps in external out-of-patient air; by a tight-body test this external band carries roughly 15-29% of total body error across models. It is largely a mask/preprocessing artifact (the MR is not zeroed inside the loose band and the sCT is saved unmasked), and every model fills it by translating residual MR into HU.

| model | external air % of error | external air MAE | MR->sCT r (air band) |
| --- | --- | --- | --- |
| U-Net | 22.03 | 92.43 | 0.72 |
| Anatomix | 23.76 | 103.13 | 0.68 |
| MAISI | 18.55 | 105.81 | 0.62 |
| cWDM | 14.92 | 97.01 | 0.71 |
| MC-DDPM | 20.78 | 107.26 | 0.61 |
| koalAI | 28.83 | 139.22 | 0.40 |

The MR->sCT r column is the pooled correlation between MR intensity and predicted HU in voxels whose ground truth is air. It is positive for every model, confirming they lift the air band toward tissue where the MR carries signal; koalAI is the weakest correlation yet the largest external error, i.e. it fills the band more uniformly rather than strictly tracking the MR.

## 3. Bone is the most under-predicted group for every model

**Claim:** bone has the most negative bias of any group for all six models, i.e. the densest tissue is always pulled down. The companion behaviour differs by model family: the regression models (U-Net, Anatomix, cWDM) over-predict air-organs and keep soft tissue near zero (the classic conditional-mean signature, both density extremes pulled toward the soft-tissue middle); MAISI, MC-DDPM and koalAI instead carry a global negative soft-tissue offset (soft and air both shifted down). So "bone undershoot" is universal, but "air overshoot, soft calibrated" is specific to the regression models.

Group bias matrix (HU, sCT - GT):

| model | bone | air-organs | soft | unlabeled |
| --- | --- | --- | --- | --- |
| U-Net | -94 | +75 | +13 | -3 |
| Anatomix | -87 | +78 | +12 | -0 |
| MAISI | -146 | -5 | -26 | +16 |
| cWDM | -175 | +78 | -25 | +11 |
| MC-DDPM | -128 | +33 | -26 | +25 |
| koalAI | -80 | +59 | -7 | +81 |

## 4. The ceiling test: dense-bone undershoot tracks the output ceiling, but bone MAE does not collapse

**Claim:** the densest cortical bone (GT > 1024 HU) is where the output ceiling bites. Capped models (U-Net/Anatomix/MAISI/cWDM) undershoot it by ~-590 HU on average; the uncapped models (MC-DDPM, koalAI) undershoot less (-351 HU). But lifting the ceiling does not solve bone: overall bone MAE is similar (capped 212 vs uncapped 193 HU; best = Anatomix), because most bone error is below the ceiling and is information-limited, not clipping-limited.

Bone summary per model (the >1024 columns isolate the ceiling effect):

| model | % body vox | bone MAE | bone bias | % body error | >1024 bias | >1024 MAE |
| --- | --- | --- | --- | --- | --- | --- |
| U-Net | 7.8 | 181.7 | -94.1 | 19.1 | -517.3 | 517.3 |
| Anatomix | 7.8 | 179.9 | -86.7 | 18.3 | -559.7 | 559.7 |
| MAISI | 7.8 | 242.2 | -146.2 | 18.7 | -676.0 | 676.0 |
| cWDM | 7.8 | 244.1 | -174.7 | 16.5 | -608.6 | 608.6 |
| MC-DDPM | 7.8 | 203.5 | -127.7 | 17.4 | -418.5 | 451.6 |
| koalAI | 7.8 | 181.6 | -80.1 | 16.6 | -282.5 | 419.9 |

**Ceiling helps only the densest sliver.** Removing the hard ceiling (MC-DDPM, koalAI) measurably reduces the >1024 HU undershoot, but that band is a small fraction of bone voxels, so overall bone MAE barely moves and bone remains the per-voxel-worst tissue for every model. This is consistent with the standing finding that bone HU is information-limited from a single MR: the cap is a secondary, model-specific aggravator on top of an intrinsic limit shared by all six.

## 5. Bone is localized, undershot and blurred for every model

**Claim:** across all models the bone failure is not mislocalization. sCT HU still ranks GT-bone above non-bone almost as well as real CT (AUC near the real-CT ceiling), and after magnitude-matching the bone edges are still softer than real CT. The error is density (bulk undershoot + edge blur), not placement.

| model | AUC real CT | AUC sCT | edge sharpness (xreal) |
| --- | --- | --- | --- |
| U-Net | 0.925 | 0.900 | 0.812 |
| Anatomix | 0.925 | 0.911 | 0.809 |
| MAISI | 0.925 | 0.847 | 0.817 |
| cWDM | 0.925 | 0.837 | 0.785 |
| MC-DDPM | 0.925 | 0.873 | 0.812 |
| koalAI | 0.925 | 0.886 | 0.869 |

## 6. Qualitative: same slice, all models

*Figure: one thorax subject, identical slice. The GT-bone outline lands on bone for every model (localized); inside it the capped models read greyer (undershoot).*

## 7. Per-model detail

The full report-10 figure set is reproduced for each model (group decomposition; within-bone error by GT density band; GT vs sCT HU calibration per CADS group; per-CADS-label error share/MAE/bias; sCT vs MR in in-mask air voxels; magnitude-matched bone-edge sharpness per region):

- U-Net (ceiling: sigmoid ~1024; body MAE 91.8)
- Anatomix (ceiling: sigmoid ~1024; body MAE 95.2)
- MAISI (ceiling: clip 1000; body MAE 125.7)
- cWDM (ceiling: clip 1024; body MAE 123.9)
- MC-DDPM (ceiling: none (>1024); body MAE 104.5)
- koalAI (ceiling: none (>1024); body MAE 104.5)

## 8. Conclusion

- **The failure structure is shared.** Severity != leverage, the ~20% external loose-mask air term, regression-to-the-mean (bone down / air up / soft flat), and located-but-undershot-and-blurred bone all hold for every one of the six models. None of these is a U-Net artifact.
- **The output ceiling is the one place models genuinely differ.** Capped models (U-Net, Anatomix, MAISI, cWDM) cannot represent GT bone above ~1000-1024 HU and undershoot it severely; MC-DDPM and koalAI, with no hard ceiling, reduce that top-band undershoot.
- **But lifting the ceiling does not fix bone.** Overall bone MAE is similar across capped and uncapped models and bone stays the per-voxel-worst tissue everywhere, because most bone error is below the ceiling and is information-limited, not clipping-limited.
- **Metric-blindness is universal.** Bone is a few percent of voxels for every model, so headline MAE/PSNR barely reflect it; a tight body mask and tissue/bone-restricted metrics are needed for any meaningful cross-model comparison.

**Reproduce:** `mm_extract.py --model M` -> `mm_mr.py --model M` (per model) -> `mm_cross.py` -> `mm_report.py`. All GT-CADS-label based, no segmentation model.
