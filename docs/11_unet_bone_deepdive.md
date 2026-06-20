# U-Net bone deep-dive: is bone the biggest problem, and why?

**Source HTML:** _html/06_unet_bone_deepdive.html
**Date:** 2026-06-19
**TL;DR:** Bone is the worst-predicted, most clinically important tissue per voxel (241 HU MAE, 4.3x soft, undershot in 100% of subjects) but NOT the biggest aggregate lever (air/soft each remove more total error). The undershoot is L1 regression to the median on an MR-ambiguous, right-skewed bone-HU distribution; MR carries almost no bone-density information, so the only true fix is adding information (bone MR sequence or prior).

Model: plain 3D U-Net (wandb `9xmodnhn`), ep799 (fully trained, center-wise split). Data: all 207 center-wise validation subjects, `full_eval_20260617`. Follow-up to report 05.

Sign convention: error = prediction - truth, so a negative bias = the model predicts too low (undershoot).

## Answers, up front (all proven below)

1. **Is bone the biggest problem?** It depends on the ruler. Per voxel and clinically, yes: bone is the worst-predicted tissue (241 HU MAE, 4.3x soft) and the only one with a large systematic undershoot. For the aggregate pixel score, no: fixing air or soft helps PSNR/MAE more than bone, because they are far more numerous (air +2.78 dB vs bone +1.29 dB).
2. **Are the big wins air AND soft (not just bone)?** Yes. Air and soft are the two biggest aggregate wins (tied on body-MAE ~13.5 HU each; air leads on PSNR, +2.78 vs +1.53 dB); bone is third (about half their size).
3. **Why does the model undershoot, and why only bone?** The L1 loss pulls every tissue toward the central (median) value given the MR. Soft sits at that center (approx unbiased), air is the low extreme (slight over-shoot), and bone is the high, skewed tail (large under-shoot). Bone is hit hardest because it is both far in the tail and MR-ambiguous.
4. **Does MR even contain bone density?** Almost none. Cortical bone looks like air on MR and trabecular bone looks like soft tissue, so a given MR brightness maps to many CT densities.

**Correctness (PASS).** Every aggregate number is reconciled against the released evaluation: the oracle baseline reproduces `body_psnr`, `body_mae_hu` and `synthrad_mae` to <0.001. All 207 subjects processed. Gates: g_base_psnr, g_base_smae, g_base_bmae, g_coverage all PASS.

## 0. How everything was measured (methods)

For each subject, four volumes are loaded on the same 1.5 mm grid: the U-Net prediction (`sample.nii.gz`, in HU), the raw ground-truth CT (`ct.nii`, full HU to ~2900), the 35-label CADS segmentation, and the body mask (`mask.nii`). All metrics are computed only over voxels inside the body mask (background outside the patient is excluded entirely).

### Tissue classes (by ground-truth HU, inside the body)

- air: HU < -300 -> lung, bowel gas, sinuses, trachea (gas inside the patient)
- soft: -300...+200 -> muscle, fat, organs, fluid
- bone: > +200 -> all skeleton; with cortical = > +1024 (dense) and mid/trabecular = 200...1024

### The metrics

- HU (Hounsfield Unit) = the CT intensity scale (air ~ -1000, water = 0, cortical bone +1000...+3000). A unit, not a metric.
- MAE = mean of |prediction - truth|, in HU. Always positive: error size, not direction.
- bias = mean of (prediction - truth), in HU. Signed: negative = undershoot. This is how we know the bone error is one-directional, not random noise.
- PSNR = 10*log10(1/MSE), in dB, higher = better. Squaring makes a few large errors dominate and the log compresses scale, which is why the biggest MAE win and biggest PSNR win need not be the same tissue.

### Clipped vs raw frame

The released validation clipped both prediction and truth to [-1024, 1024] (what the model was trained and scored on). Reported in that clipped frame by default (matches leaderboard), with the raw frame (true HU) noted where it matters for bone.

### The oracle counterfactual

To ask "how much would perfect tissue X help?", copy the ground-truth HU into the prediction only inside that tissue, leave everything else untouched, and recompute the exact reported metric. The improvement is exactly the error currently sitting in that tissue. Done for air / soft / bone / cortical / skull.

### Regression-to-mean calibration (Section 5)

Accumulate a 2D histogram of (true bone HU, predicted HU) over all bone voxels, then for each true-HU bin read off the mean predicted HU. A perfect model would lie on the identity line.

### MR-information test (Section 6)

MR intensity is normalized per-volume (and bias-field augmented), so absolute MR values are not comparable across subjects. MR intensity is ranked within each subject's body (0 = darkest, 1 = brightest) and related to CT HU per tissue (rank-based / scale-invariant).

## 1. Is bone the biggest problem? It depends on the ruler

### Per voxel: bone is clearly worst

| tissue | % of body voxels | per-voxel MAE (HU) | signed bias (HU) | % of total error |
| --- | --- | --- | --- | --- |
| air | 27.2 | 122.6 | 46.5 | 41.0 |
| soft | 68.1 | 55.7 | 14.2 | 43.7 |
| bone | 4.7 | 240.8 | -179.6 | 15.4 |

Bone has the highest per-voxel MAE (241 HU) and the only large systematic bias (-180 HU undershoot). Soft is best per voxel (56 HU, near-zero bias).

### Aggregate (leaderboard): air and soft win, bone is third

The oracle gain = the error currently held by each tissue. Because air (27%) and soft (68%) hold far more voxels, they hold more total error than rare bone (5%), even though each bone voxel is worse:

| fix this tissue -> | ΔPSNR (dB) | Δ body-MAE (HU) | Δ full-HU MAE (HU) |
| --- | --- | --- | --- |
| air | 2.78 | 13.51 | 34.24 |
| soft | 1.53 | 13.37 | 34.81 |
| bone | 1.29 | 7.22 | 22.77 |
| cortical | 0.28 | 1.68 | 8.29 |
| skull | 0.42 | 3.42 | 11.64 |

Air and soft are the biggest aggregate wins (tied on body-MAE; air leads on PSNR); bone is about half their size; cortical/skull move the clipped score the least (the clip hides their error). "Biggest" depends on whether you weight per-voxel/clinical accuracy (bone) or the aggregate pixel metric (air/soft).

## 2. Why "air" is real intra-body gas, not the background

Objection: "isn't air just the easy black background?" No. Two proofs:

(i) **The background is excluded.** For a representative thorax subject (`1THB006`), of the 20.3 M total voxels, 15.4 M air voxels lie OUTSIDE the body and are never scored; only 2.05 M air voxels INSIDE the body count. The body mask is just 24% of the volume. The scored air has mean HU -866 (lung-like), not the -1024 of background.

(ii) **Air% tracks anatomy.** If "air" were background it would be constant; instead it follows where gas actually is (thorax lungs highest):

| region | air % of body | air per-voxel MAE (HU) |
| --- | --- | --- |
| brain | 32 | 151 |
| head_neck | 28 | 133 |
| thorax | 36 | 86 |
| abdomen | 24 | 112 |
| pelvis | 22 | 113 |

## 3. Is the undershoot universal? Yes

Across all 207 subjects, 100% undershoot bone and 100% undershoot cortical bone; within each subject a mean of 81% of bone voxels are predicted too low. This is the model's default behaviour, not a few outliers.

## 4. Do we see it? Yes

One representative subject per region (median bone-MAE). The predicted skull/bone is visibly grey (undershot) vs the bright cortical GT in the same window; the dense-bone error is the clipped error the metric never sees (brain, head_neck, thorax, abdomen, pelvis all show predicted cortical bone grey/undershot, with a final panel showing the hidden clipped-away error).

## 5. Why the model undershoots, and why ONLY bone

The L1 loss is minimized by predicting the median of the true HU values consistent with a given MR input (L2 -> the mean). When one MR appearance maps to many possible HU (Section 6), the model can only emit one number, so it emits that central value. Not a bug: the loss hedging.

This pull acts on every tissue, but its effect depends on where the tissue sits in the HU distribution. From the bias column in Section 1: air (low extreme) is pulled up (bias +47 HU, overshoot); soft (the center) barely moves (+14 HU); bone (high, skewed tail) is pulled down (-180 HU, undershoot; cortical -521 HU). Same mechanism, opposite directions.

Two conditions make the bone error large and systematic, and only bone meets both: (a) the MR is ambiguous about its HU (wide conditional spread) and (b) it sits in a skewed tail (so the central compromise is one-directional). Soft fails both (narrow, central) -> tiny unbiased error. Bone meets both -> large undershoot.

*Figure: predicted bone HU collapses toward the mean and stops ~951 HU, BELOW the 1024 sigmoid cap, so the architecture cap is not the limiting factor; regression to the mean is.*

## 6. Does MR contain bone-density information? Almost none

MR signal comes from mobile hydrogen protons (water/fat). Cortical bone has almost none -> a dark void that looks like air. Trabecular bone shows its marrow (fat/water), so it looks like soft tissue, not like its true high HU. MR brightness does not encode mineral density. Proof on this data: bone's MR-brightness distribution overlaps soft tissue by 0.78 and cortical overlaps air by 0.36 (1.0 = indistinguishable). A given MR brightness maps to many CT densities.

Quantitatively, knowing the MR removes only 12% of the bone-HU spread (std 348->307 HU; for soft it removes 3%): MR barely narrows uncertainty about a bone voxel's HU. (The squared rank correlation between MR and HU, a coarser rank-based measure, is also low in bone: 0.14 vs 0.20 for soft.) Bone HU is also intrinsically ~4x wider than soft (std 281 vs 73 HU): a wide target the MR cannot resolve is exactly what an L1 model collapses to the mean. (Consistent with the prior cross-model result that even the uncapped diffusion baselines undershoot bone.)

## 7. Supporting causes

### Bone is rare -> negligible weight in the uniform loss

Bone is 5% of body voxels under a uniform L1, so it contributes little gradient, even though it holds 15% of the error in the clipped frame the loss actually optimizes (3.3x its voxel share; 17% in the raw frame).

### It is a density-magnitude failure, not a localization failure

The model mostly knows where bone is (shape Dice 0.61); the error is the HU magnitude in the interior (interior MAE 330 > boundary 222 HU). In thin-bone body regions it additionally under-detects bone (missed 40% fall below the 200-HU threshold because they are undershot).

| region | shape Dice | missed frac | FP frac | interior MAE | boundary MAE |
| --- | --- | --- | --- | --- | --- |
| brain | 0.79 | 0.16 | 0.25 | 350.51 | 277.10 |
| head_neck | 0.64 | 0.34 | 0.37 | 381.69 | 244.51 |
| thorax | 0.47 | 0.59 | 0.45 | 317.31 | 199.86 |
| abdomen | 0.45 | 0.61 | 0.47 | 337.81 | 201.18 |
| pelvis | 0.67 | 0.40 | 0.24 | 237.03 | 147.91 |

## 8. How to fix it

- **Add the missing information (highest ceiling).** A bone-sensitive MR sequence (UTE/ZTE) that actually images cortical bone, or inject a prior (population CT/bone atlas, paired-anatomy retrieval). The undershoot is fundamentally a missing-information problem, so this is the only route to true per-voxel density.
- **Stop the loss hedging (medium ceiling).** A generative/probabilistic model (GAN, diffusion, distributional or quantile loss) emits a sharp plausible bone value instead of the blurry mean; or bone-weighted loss to fix the rarity. These reduce bias and sharpen output but cannot recover information the MR lacks.
- **Work around it (most pragmatic).** Bulk-density override (segment bone, assign a standard density) and evaluate on the clinical task (dose/DVH, bone-specific metrics) rather than clipped PSNR/MAE, which hide this error.

**Honest ceiling.** The undershoot is the L1 loss behaving optimally on an input that is genuinely ambiguous about bone density. You can sharpen it, de-bias it, or sidestep it, but the only way to truly recover dense cortical HU is to add information the standard MR does not contain.

## 9. Conclusion

- Bone is the worst-predicted and most clinically important tissue per voxel (4.3x soft), undershot in 100% of subjects.
- It is not the biggest aggregate lever: air and soft each remove more total error (air +2.78 dB PSNR vs bone +1.29); whether bone is "the biggest problem" depends on per-voxel/clinical vs leaderboard.
- The mechanism is L1 regression to the median on an MR-ambiguous, right-skewed bone-HU distribution: every tissue is pulled to its conditional center, and bone (the wide, skewed, MR-unresolvable tail) is undershot hardest. The sigmoid cap is secondary (predictions stop below it).
- The fix is information (a bone MR sequence or a prior), with generative/bone-weighted losses and bulk-density overrides as partial mitigations; and bone-aware evaluation, since clipped pixel metrics hide the error.

## Reproducibility

Scripts in `src/evaluate/unet_failure/`: `bone_extract.py` (per-subject oracle, biases, loc/mag, MR Spearman), `bone_aggregate.py` (rollups + gates + per-tissue table + air proof), `mr_tissue.py` (MR-rank by tissue), `bone_figures.py`, `bone_report.py`; `run_all.py` runs the whole chain. Data in `evaluation_results/unet_failure_20260619/`. Oracle replicates `compute_metrics_body`; GT = raw `ct.nii` in canonical RAS; MR analysis uses within-subject rank.
